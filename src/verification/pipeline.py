"""Verification pipeline orchestrator."""

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from .confidence import ConfidenceCalibrator
from .cove import ChainOfVerification
from .critic import CRITICVerifier, HighStakesDetector
from .hhem import HHEMScorer
from .metrics import VerificationMetricsTracker
from .models import (
    BatchVerificationResult,
    ContradictionDetail,
    VerificationConfig,
    VerificationMethod,
    VerificationResult,
    VerificationStatus,
)

if TYPE_CHECKING:
    from ..knowledge.graph import IncrementalKnowledgeGraph
    from ..models.findings import Finding


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
        search_callback: Callable[[str], Any] | None = None,
        config: VerificationConfig | None = None,
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
        self.hhem = HHEMScorer() if self.config.enable_hhem else None

    async def verify_intern_finding(
        self,
        finding: "Finding",
        session_id: str,
        source_content: str | None = None,
    ) -> VerificationResult:
        """Streaming verification when intern submits a finding.

        Fast, lightweight verification for real-time use.
        Target latency: <500ms

        Args:
            finding: The finding to verify
            session_id: Current session ID
            source_content: Original source text the finding was extracted from

        Returns:
            VerificationResult with calibrated confidence
        """
        finding_id = str(finding.id or hash(finding.content))

        if not self.config.enable_streaming_verification:
            return VerificationResult(
                finding_id=finding_id,
                original_confidence=finding.confidence,
                verified_confidence=finding.confidence,
                verification_status=VerificationStatus.SKIPPED,
                verification_method=VerificationMethod.STREAMING,
            )

        # HHEM source-grounding check (runs before CoVe to enable early reject)
        hhem_score = -1.0
        if self.hhem and source_content:
            hhem_score = await self.hhem.score(source_content, finding.content)

            # Early reject if clearly not grounded in source
            if 0 <= hhem_score < self.config.hhem_reject_threshold:
                result = VerificationResult(
                    finding_id=finding_id,
                    original_confidence=finding.confidence,
                    verified_confidence=max(0.0, finding.confidence - 0.15),
                    verification_status=VerificationStatus.REJECTED,
                    verification_method=VerificationMethod.STREAMING,
                    hhem_grounding_score=hhem_score,
                )
                await self.metrics.record_result(result)
                return result

        # Quick KG match if available
        kg_support_score = 0.0
        kg_entity_matches = 0
        kg_supporting_relations = 0
        if self.knowledge_graph and self.config.enable_kg_verification:
            kg_support_score, kg_entity_matches, kg_supporting_relations = (
                await self._get_kg_support(finding)
            )

        # Run streaming CoVe
        result = await self.cove.verify_streaming(
            finding_content=finding.content,
            finding_id=finding_id,
            original_confidence=finding.confidence,
            source_url=finding.source_url,
            search_query=finding.search_query,
        )

        # Attach HHEM and KG scores
        result.hhem_grounding_score = hhem_score
        result.kg_support_score = kg_support_score
        result.kg_entity_matches = kg_entity_matches
        result.kg_supporting_relations = kg_supporting_relations

        # Recalibrate with all signals (KG + HHEM + CoVe)
        if kg_support_score > 0 or hhem_score >= 0:
            # Pass -1 for CoVe consistency if CoVe was skipped (parsing failed),
            # so the calibrator treats it as "no data" rather than a negative signal
            cove_score = (
                result.consistency_score
                if result.verification_status != VerificationStatus.SKIPPED
                else -1.0
            )
            calibration = self.calibrator.calibrate(
                original_confidence=finding.confidence,
                cove_consistency_score=cove_score,
                kg_support_score=kg_support_score,
                hhem_grounding_score=hhem_score,
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
            batch = findings[i : i + self.config.parallel_batch_size]
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
        tasks = [self._verify_single_batch(finding, session_id) for finding in findings]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions - convert to skipped results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create a fallback result for failed verification
                finding = findings[i]
                processed_results.append(
                    VerificationResult(
                        finding_id=str(finding.id or hash(finding.content)),
                        original_confidence=finding.confidence,
                        verified_confidence=finding.confidence,
                        verification_status=VerificationStatus.SKIPPED,
                        verification_method=VerificationMethod.BATCH,
                        error=str(result),
                    )
                )
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
        kg_entity_matches = 0
        kg_supporting_relations = 0
        has_contradictions = False
        contradictions = []

        if self.knowledge_graph and self.config.enable_kg_verification:
            kg_support_score, kg_entity_matches, kg_supporting_relations = (
                await self._get_kg_support(finding)
            )
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

        # Attach KG counts to CoVe result
        cove_result.kg_entity_matches = kg_entity_matches
        cove_result.kg_supporting_relations = kg_supporting_relations

        # Check if CRITIC is needed:
        # - Low confidence OR contradictions found, AND high-stakes content
        use_critic = (
            self.config.enable_critic
            and (
                cove_result.verified_confidence < self.config.critic_confidence_threshold
                or has_contradictions
            )
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
            critic_result.kg_entity_matches = kg_entity_matches
            critic_result.kg_supporting_relations = kg_supporting_relations
            return critic_result

        # Add contradictions to CoVe result
        cove_result.contradictions = contradictions
        return cove_result

    async def _get_kg_support(self, finding: "Finding") -> tuple[float, int, int]:
        """Get KG support score and entity/relation counts for a finding.

        Returns:
            Tuple of (score, entity_matches, supporting_relations)
        """
        if not self.knowledge_graph:
            return 0.0, 0, 0

        try:
            return await self.knowledge_graph.get_kg_support_score(
                content=finding.content,
                source_url=finding.source_url,
            )
        except Exception:
            return 0.0, 0, 0

    async def _check_kg_contradictions(self, finding: "Finding") -> dict:
        """Check for contradictions in the KG."""
        if not self.knowledge_graph:
            return {"has_contradictions": False, "contradictions": []}

        try:
            return await self.knowledge_graph.check_contradictions_detailed(
                content=finding.content,
            )
        except Exception:
            return {"has_contradictions": False, "contradictions": []}

    def get_metrics_summary(self) -> dict:
        """Get summary of verification metrics."""
        return self.metrics.get_summary()

    def get_metrics_report_section(self) -> str:
        """Get markdown section for report."""
        return self.metrics.get_report_section()

    def reset_metrics(self, session_id: str | None = None) -> None:
        """Reset metrics for a new session."""
        self.metrics.reset()
        self.metrics.session_id = session_id


def create_verification_pipeline(
    llm_callback: Callable[[str, str], Any],
    knowledge_graph: Optional["IncrementalKnowledgeGraph"] = None,
    search_callback: Callable[[str], Any] | None = None,
    config: VerificationConfig | None = None,
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
