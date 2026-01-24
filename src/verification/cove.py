"""Chain-of-Verification (CoVe) implementation.

CoVe is a 4-step verification process:
1. Generate baseline response (already have the finding)
2. Plan verification questions
3. Execute independent verification (answer questions without seeing original)
4. Generate final verified response

Reference: "Chain-of-Verification Reduces Hallucination in Large Language Models"
"""

import asyncio
import json
import re
import time
from typing import Callable, Optional, Any

from .models import (
    VerificationQuestion,
    VerificationResult,
    VerificationStatus,
    VerificationMethod,
    VerificationConfig,
)
from .confidence import ConfidenceCalibrator


class ChainOfVerification:
    """Chain-of-Verification for fact verification.

    Implements a 4-step verification process:
    1. Baseline: The original finding (already have this)
    2. Plan: Generate verification questions
    3. Execute: Answer questions independently
    4. Verify: Check consistency and calibrate confidence
    """

    def __init__(
        self,
        llm_callback: Callable[[str, str], Any],  # (prompt, model) -> response
        config: Optional[VerificationConfig] = None,
    ):
        """Initialize CoVe verifier.

        Args:
            llm_callback: Async function to call LLM with (prompt, model)
            config: Verification configuration
        """
        self.llm_callback = llm_callback
        self.config = config or VerificationConfig()
        self.calibrator = ConfidenceCalibrator(config)

    async def verify_streaming(
        self,
        finding_content: str,
        finding_id: str,
        original_confidence: float,
        source_url: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> VerificationResult:
        """Streaming verification - fast, lightweight for real-time use.

        Target latency: <500ms
        Uses 1-2 questions and Haiku model.

        Args:
            finding_content: The content of the finding to verify
            finding_id: Unique identifier for the finding
            original_confidence: Original confidence score (0-1)
            source_url: Optional source URL for context
            search_query: Optional search query that produced this finding

        Returns:
            VerificationResult with calibrated confidence
        """
        start_time = time.time()

        try:
            # Step 1: Generate 1-2 quick verification questions
            questions = await self._generate_questions_streaming(
                finding_content, source_url, search_query
            )

            if not questions:
                # Fallback: no verification possible
                return VerificationResult(
                    finding_id=finding_id,
                    original_confidence=original_confidence,
                    verified_confidence=original_confidence,
                    verification_status=VerificationStatus.SKIPPED,
                    verification_method=VerificationMethod.STREAMING,
                    verification_time_ms=(time.time() - start_time) * 1000,
                    error="Could not generate verification questions",
                )

            # Step 2: Answer questions independently (parallel for speed)
            answered_questions = await self._answer_questions_parallel(
                questions, finding_content
            )

            # Step 3: Calculate consistency and calibrate
            consistency_score = self._calculate_consistency(answered_questions)

            calibration = self.calibrator.calibrate(
                original_confidence=original_confidence,
                cove_consistency_score=consistency_score,
            )

            return VerificationResult(
                finding_id=finding_id,
                original_confidence=original_confidence,
                verified_confidence=calibration.calibrated_confidence,
                verification_status=calibration.status,
                verification_method=VerificationMethod.STREAMING,
                questions_asked=answered_questions,
                consistency_score=consistency_score,
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return VerificationResult(
                finding_id=finding_id,
                original_confidence=original_confidence,
                verified_confidence=original_confidence,
                verification_status=VerificationStatus.SKIPPED,
                verification_method=VerificationMethod.STREAMING,
                verification_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def verify_batch(
        self,
        finding_content: str,
        finding_id: str,
        original_confidence: float,
        source_url: Optional[str] = None,
        search_query: Optional[str] = None,
        kg_support_score: float = 0.0,
        has_contradictions: bool = False,
    ) -> VerificationResult:
        """Batch verification - comprehensive, thorough analysis.

        Target latency: ~2s per finding
        Uses 3-5 questions and Sonnet model.

        Args:
            finding_content: The content of the finding to verify
            finding_id: Unique identifier for the finding
            original_confidence: Original confidence score (0-1)
            source_url: Optional source URL for context
            search_query: Optional search query that produced this finding
            kg_support_score: KG corroboration score (0-1)
            has_contradictions: Whether KG found contradictions

        Returns:
            VerificationResult with calibrated confidence
        """
        start_time = time.time()

        try:
            # Step 1: Generate 3-5 comprehensive verification questions
            questions = await self._generate_questions_batch(
                finding_content, source_url, search_query
            )

            if not questions:
                return VerificationResult(
                    finding_id=finding_id,
                    original_confidence=original_confidence,
                    verified_confidence=original_confidence,
                    verification_status=VerificationStatus.SKIPPED,
                    verification_method=VerificationMethod.BATCH,
                    verification_time_ms=(time.time() - start_time) * 1000,
                    error="Could not generate verification questions",
                )

            # Step 2: Answer questions independently
            answered_questions = await self._answer_questions_parallel(
                questions, finding_content, use_batch_model=True
            )

            # Step 3: Calculate consistency
            consistency_score = self._calculate_consistency(answered_questions)

            # Step 4: Calibrate with all signals
            calibration = self.calibrator.calibrate(
                original_confidence=original_confidence,
                cove_consistency_score=consistency_score,
                kg_support_score=kg_support_score,
                has_contradictions=has_contradictions,
            )

            return VerificationResult(
                finding_id=finding_id,
                original_confidence=original_confidence,
                verified_confidence=calibration.calibrated_confidence,
                verification_status=calibration.status,
                verification_method=VerificationMethod.BATCH,
                questions_asked=answered_questions,
                consistency_score=consistency_score,
                kg_support_score=kg_support_score,
                verification_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return VerificationResult(
                finding_id=finding_id,
                original_confidence=original_confidence,
                verified_confidence=original_confidence,
                verification_status=VerificationStatus.SKIPPED,
                verification_method=VerificationMethod.BATCH,
                verification_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def _generate_questions_streaming(
        self,
        finding_content: str,
        source_url: Optional[str],
        search_query: Optional[str],
    ) -> list[VerificationQuestion]:
        """Generate 1-2 quick verification questions."""
        context = f"Source: {source_url}" if source_url else ""
        if search_query:
            context += f"\nSearch query: {search_query}"

        prompt = f"""Generate 1-2 quick verification questions for this claim.

CLAIM: {finding_content}
{context}

Questions should check the most critical facts. Return as JSON:
[
  {{"question": "Is X true?", "aspect": "factual"}},
  {{"question": "When did Y happen?", "aspect": "temporal"}}
]

Aspects: factual, temporal, source, quantitative, causal
Return ONLY the JSON array."""

        response = await self.llm_callback(prompt, self.config.streaming_model)
        return self._parse_questions(response)

    async def _generate_questions_batch(
        self,
        finding_content: str,
        source_url: Optional[str],
        search_query: Optional[str],
    ) -> list[VerificationQuestion]:
        """Generate 3-5 comprehensive verification questions."""
        context = f"Source: {source_url}" if source_url else ""
        if search_query:
            context += f"\nSearch query: {search_query}"

        prompt = f"""Generate 3-5 verification questions to thoroughly check this claim.

CLAIM: {finding_content}
{context}

Cover different verification aspects:
- Factual accuracy (are the stated facts correct?)
- Temporal accuracy (are dates/timeframes correct?)
- Source attribution (is the source reliable?)
- Quantitative claims (are numbers accurate?)
- Causal claims (are cause-effect relationships valid?)

Return as JSON array:
[
  {{"question": "Specific question?", "aspect": "factual"}},
  {{"question": "Another question?", "aspect": "temporal"}}
]

Return ONLY the JSON array, no explanation."""

        response = await self.llm_callback(prompt, self.config.batch_model)
        return self._parse_questions(response)

    async def _answer_questions_parallel(
        self,
        questions: list[VerificationQuestion],
        original_finding: str,
        use_batch_model: bool = False,
    ) -> list[VerificationQuestion]:
        """Answer verification questions independently in parallel."""
        model = self.config.batch_model if use_batch_model else self.config.streaming_model

        async def answer_question(q: VerificationQuestion) -> VerificationQuestion:
            # Importantly: don't show the original finding to avoid bias
            prompt = f"""Answer this verification question based on your knowledge.

QUESTION: {q.question}
ASPECT: {q.aspect}

Provide:
1. Your answer to the question
2. Confidence (0.0-1.0)
3. Whether this supports or contradicts the claim being verified

Return as JSON:
{{"answer": "Your answer", "confidence": 0.8, "supports": true}}

Return ONLY the JSON."""

            try:
                response = await self.llm_callback(prompt, model)
                data = self._parse_json(response)
                if data:
                    q.independent_answer = data.get("answer", "")
                    q.confidence = data.get("confidence", 0.5)
                    q.supports_original = data.get("supports", None)
            except Exception:
                q.independent_answer = None
                q.confidence = 0.0
                q.supports_original = None

            return q

        # Run all questions in parallel
        answered = await asyncio.gather(*[answer_question(q) for q in questions])
        return list(answered)

    def _calculate_consistency(
        self, questions: list[VerificationQuestion]
    ) -> float:
        """Calculate consistency score from answered questions.

        Returns 0-1 score based on:
        - How many questions support the original finding
        - Average confidence of answers
        """
        if not questions:
            return 0.0

        supporting = 0
        total_confidence = 0.0
        answered_count = 0

        for q in questions:
            if q.independent_answer is not None:
                answered_count += 1
                total_confidence += q.confidence
                if q.supports_original:
                    supporting += 1

        if answered_count == 0:
            return 0.0

        # Consistency = (support_ratio * 0.6) + (avg_confidence * 0.4)
        support_ratio = supporting / answered_count
        avg_confidence = total_confidence / answered_count

        return (support_ratio * 0.6) + (avg_confidence * 0.4)

    def _parse_questions(self, response: str) -> list[VerificationQuestion]:
        """Parse verification questions from LLM response."""
        questions = []
        try:
            # Find JSON array in response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                for item in data:
                    if isinstance(item, dict) and item.get("question"):
                        questions.append(VerificationQuestion(
                            question=item["question"],
                            aspect=item.get("aspect", "factual"),
                        ))
        except (json.JSONDecodeError, KeyError):
            pass

        # Limit to configured max
        max_questions = self.config.max_cove_questions_batch
        return questions[:max_questions]

    def _parse_json(self, response: str) -> Optional[dict]:
        """Parse JSON object from LLM response."""
        try:
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, KeyError):
            pass
        return None
