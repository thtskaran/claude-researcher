"""Verification pipeline orchestrator."""

import asyncio
import time
from typing import Callable, Optional, Any, TYPE_CHECKING

from .models import (
    VerificationResult,
    VerificationStatus,
    VerificationMethod,
    VerificationConfig,
    BatchVerificationResult,
    ContradictionDetail,
)
from .cove import ChainOfVerification
from .critic import CRITICVerifier, HighStakesDetector
from .confidence import ConfidenceCalibrator
from .metrics import VerificationMetricsTracker

if TYPE_CHECKING:
    from ..models.findings import Finding
    from ..knowledge.graph import IncrementalKnowledgeGraph


class VerificationPipeline:
    """Main orchestrator for the verification system.

    Coordinates:
    - Streaming verification for real-time finding submission
    - Batch verification during synthesis
    - KG integration for corroboration
    - CRITIC for high-stakes findings
    """

    def __init__(
        self,
        llm_callback: Callable[[str, str], Any],  # (prompt, model) -> response
        knowledge_graph: Optional["IncrementalKnowledgeGraph"] = None,
        search_callback: Optional[Callable[[str], Any]] = None,
        config: Optional[VerificationConfig] = None,
    ):
        """Initialize the verification pipeline.

        Args:
            llm_callback: Async function to call LLM with (prompt, model)
            knowledge_graph: Optional KG for corroboration
            search_callback: Optional async function for web search (CRITIC)
            config: Verification configuration
        """
        self.config = config or VerificationConfig()
        self.llm_callback = llm_callback
        self.knowledge_graph = knowledge_graph
        self.search_callback = search_callback

        # Initialize components
        self.cove = ChainOfVerification(llm_callback, self.config)
        self.critic = CRITICVerifier(llm_callback, search_callback, self.config)
        self.calibrator = ConfidenceCalibrator(self.config)
        self.high_stakes_detector = HighStakesDetector()
        self.metrics = VerificationMetricsTracker()

    async def verify_intern_finding(
        self,
        finding: "Finding",
        session_id: str,
    ) -> VerificationResult:
        """Streaming verification when intern submits a finding.

        Fast, lightweight verification for real-time use.
        Target latency: <500ms

        Args:
            finding: The finding to verify
            session_id: Current session ID

        Returns:
            VerificationResult with calibrated confidence
        """
        if not self.config.enable_streaming_verification:
            return VerificationResult(
                finding_id=str(finding.id or hash(finding.content)),
                original_confidence=finding.confidence,
                verified_confidence=finding.confidence,
                verification_status=VerificationStatus.SKIPPED,
                verification_method=VerificationMethod.STREAMING,
            )

        # Quick KG match if available
        kg_support_score = 0.0
        if self.knowledge_graph and self.config.enable_kg_verification:
            kg_support_score = await self._get_kg_support(finding)

        # Run streaming CoVe
        result = await self.cove.verify_streaming(
            finding_content=finding.content,
            finding_id=str(finding.id or hash(finding.content)),
            original_confidence=finding.confidence,
            source_url=finding.source_url,
            search_query=finding.search_query,
        )

        # Apply KG boost if matched
        if kg_support_score > 0:
            result.kg_support_score = kg_support_score
            # Recalibrate with KG
            calibration = self.calibrator.calibrate(
                original_confidence=finding.confidence,
                cove_consistency_score=result.consistency_score,
                kg_support_score=kg_support_score,
            )
            result.verified_confidence = calibration.calibrated_confidence
            result.verification_status = calibration.status

        # Track metrics
        await self.metrics.record_result(result)

        return result

    async def verify_batch(
        self,
        findings: list["Finding"],
        session_id: str,
    ) -> BatchVerificationResult:
        """Batch verification during manager synthesis.

        Comprehensive verification with KG consistency checks and CRITIC for high-stakes.
        Target latency: ~2s per finding average.

        Args:
            findings: List of findings to verify
            session_id: Current session ID

        Returns:
            BatchVerificationResult with all verification results
        """
        start_time = time.time()

        if not self.config.enable_batch_verification:
            return BatchVerificationResult(
                session_id=session_id,
                total_findings=len(findings),
                verified_count=0,
                flagged_count=0,
                rejected_count=0,
                skipped_count=len(findings),
            )

        # Process in parallel batches
        results = []
        all_contradictions = []

        for i in range(0, len(findings), self.config.parallel_batch_size):
            batch = findings[i:i + self.config.parallel_batch_size]
            batch_results = await self._verify_batch_parallel(batch, session_id)
            results.extend(batch_results)

        # Collect contradictions
        for result in results:
            all_contradictions.extend(result.contradictions)

        # Count by status
        verified = sum(1 for r in results if r.verification_status == VerificationStatus.VERIFIED)
        flagged = sum(1 for r in results if r.verification_status == VerificationStatus.FLAGGED)
        rejected = sum(1 for r in results if r.verification_status == VerificationStatus.REJECTED)
        skipped = sum(1 for r in results if r.verification_status == VerificationStatus.SKIPPED)

        total_time = (time.time() - start_time) * 1000

        batch_result = BatchVerificationResult(
            session_id=session_id,
            total_findings=len(findings),
            verified_count=verified,
            flagged_count=flagged,
            rejected_count=rejected,
            skipped_count=skipped,
            results=results,
            contradictions_found=all_contradictions,
            total_time_ms=total_time,
            avg_time_per_finding_ms=total_time / len(findings) if findings else 0,
        )

        # Track metrics
        await self.metrics.record_batch(batch_result)

        return batch_result

    async def _verify_batch_parallel(
        self,
        findings: list["Finding"],
        session_id: str,
    ) -> list[VerificationResult]:
        """Verify a batch of findings in parallel."""
        tasks = [
            self._verify_single_batch(finding, session_id)
            for finding in findings
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions - convert to skipped results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create a fallback result for failed verification
                finding = findings[i]
                processed_results.append(VerificationResult(
                    finding_id=str(finding.id or hash(finding.content)),
                    original_confidence=finding.confidence,
                    verified_confidence=finding.confidence,
                    verification_status=VerificationStatus.SKIPPED,
                    verification_method=VerificationMethod.BATCH,
                    error=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _verify_single_batch(
        self,
        finding: "Finding",
        session_id: str,
    ) -> VerificationResult:
        """Verify a single finding in batch mode."""
        finding_id = str(finding.id or hash(finding.content))

        # Get KG signals
        kg_support_score = 0.0
        has_contradictions = False
        contradictions = []

        if self.knowledge_graph and self.config.enable_kg_verification:
            kg_support_score = await self._get_kg_support(finding)
            contradiction_result = await self._check_kg_contradictions(finding)
            has_contradictions = contradiction_result.get("has_contradictions", False)
            if has_contradictions:
                contradictions = [
                    ContradictionDetail(
                        finding_id=finding_id,
                        conflicting_finding_id=c.get("conflicting_id", "unknown"),
                        description=c.get("description", "Contradiction detected"),
                        severity=c.get("severity", "medium"),
                    )
                    for c in contradiction_result.get("contradictions", [])
                ]

        # Run batch CoVe
        cove_result = await self.cove.verify_batch(
            finding_content=finding.content,
            finding_id=finding_id,
            original_confidence=finding.confidence,
            source_url=finding.source_url,
            search_query=finding.search_query,
            kg_support_score=kg_support_score,
            has_contradictions=has_contradictions,
        )

        # Check if CRITIC is needed (low confidence + high stakes)
        use_critic = (
            self.config.enable_critic
            and cove_result.verified_confidence < self.config.critic_confidence_threshold
            and self.high_stakes_detector.is_high_stakes(finding.content)
        )

        if use_critic:
            critic_result = await self.critic.verify(
                finding_content=finding.content,
                finding_id=finding_id,
                original_confidence=finding.confidence,
                source_url=finding.source_url,
                cove_result=cove_result,
            )
            critic_result.contradictions = contradictions
            return critic_result

        # Add contradictions to CoVe result
        cove_result.contradictions = contradictions
        return cove_result

    async def _get_kg_support(self, finding: "Finding") -> float:
        """Get KG support score for a finding."""
        if not self.knowledge_graph:
            return 0.0

        try:
            # Use the KG's support scoring method
            if hasattr(self.knowledge_graph, 'get_kg_support_score'):
                return await self.knowledge_graph.get_kg_support_score(
                    content=finding.content,
                    source_url=finding.source_url,
                )
            return 0.0
        except Exception:
            return 0.0

    async def _check_kg_contradictions(self, finding: "Finding") -> dict:
        """Check for contradictions in the KG."""
        if not self.knowledge_graph:
            return {"has_contradictions": False, "contradictions": []}

        try:
            if hasattr(self.knowledge_graph, 'check_contradictions_detailed'):
                return await self.knowledge_graph.check_contradictions_detailed(
                    content=finding.content,
                )
            return {"has_contradictions": False, "contradictions": []}
        except Exception:
            return {"has_contradictions": False, "contradictions": []}

    def get_metrics_summary(self) -> dict:
        """Get summary of verification metrics."""
        return self.metrics.get_summary()

    def get_metrics_report_section(self) -> str:
        """Get markdown section for report."""
        return self.metrics.get_report_section()

    def reset_metrics(self, session_id: Optional[str] = None) -> None:
        """Reset metrics for a new session."""
        self.metrics.reset()
        self.metrics.session_id = session_id


def create_verification_pipeline(
    llm_callback: Callable[[str, str], Any],
    knowledge_graph: Optional["IncrementalKnowledgeGraph"] = None,
    search_callback: Optional[Callable[[str], Any]] = None,
    config: Optional[VerificationConfig] = None,
) -> VerificationPipeline:
    """Factory function to create a verification pipeline.

    Args:
        llm_callback: Async function to call LLM with (prompt, model)
        knowledge_graph: Optional KG for corroboration
        search_callback: Optional async function for web search
        config: Verification configuration

    Returns:
        Configured VerificationPipeline
    """
    return VerificationPipeline(
        llm_callback=llm_callback,
        knowledge_graph=knowledge_graph,
        search_callback=search_callback,
        config=config,
    )
