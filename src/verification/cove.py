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
from collections.abc import Callable

from .confidence import ConfidenceCalibrator
from .models import (
    VerificationConfig,
    VerificationMethod,
    VerificationQuestion,
    VerificationResult,
    VerificationStatus,
)

# JSON schemas for structured output
QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "aspect": {
                        "type": "string",
                        "enum": [
                            "factual", "temporal", "source",
                            "quantitative", "causal",
                        ],
                    },
                },
                "required": ["question", "aspect"],
            },
        },
    },
    "required": ["questions"],
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "supports": {"type": "boolean"},
    },
    "required": ["answer", "confidence", "supports"],
}


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
        llm_callback: Callable,  # (prompt, model, **kwargs) -> response
        config: VerificationConfig | None = None,
    ):
        """Initialize CoVe verifier.

        Args:
            llm_callback: Async function to call LLM. Accepts (prompt, model)
                and optionally output_format kwarg for structured output.
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
        source_url: str | None = None,
        search_query: str | None = None,
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
        source_url: str | None = None,
        search_query: str | None = None,
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
        source_url: str | None,
        search_query: str | None,
    ) -> list[VerificationQuestion]:
        """Generate 1-2 quick verification questions."""
        context = f"Source: {source_url}" if source_url else ""
        if search_query:
            context += f"\nSearch query: {search_query}"

        prompt = (
            f"Generate 1-2 quick verification questions for this claim.\n\n"
            f"CLAIM: {finding_content}\n{context}\n\n"
            f"Questions should check the most critical facts.\n"
            f"Aspects: factual, temporal, source, quantitative, causal"
        )

        return await self._call_for_questions(
            prompt, self.config.streaming_model,
            max_q=self.config.max_cove_questions_streaming,
        )

    async def _generate_questions_batch(
        self,
        finding_content: str,
        source_url: str | None,
        search_query: str | None,
    ) -> list[VerificationQuestion]:
        """Generate 3-5 comprehensive verification questions."""
        context = f"Source: {source_url}" if source_url else ""
        if search_query:
            context += f"\nSearch query: {search_query}"

        prompt = (
            f"Generate 3-5 verification questions to thoroughly check "
            f"this claim.\n\n"
            f"CLAIM: {finding_content}\n{context}\n\n"
            f"Cover different verification aspects:\n"
            f"- Factual accuracy (are the stated facts correct?)\n"
            f"- Temporal accuracy (are dates/timeframes correct?)\n"
            f"- Source attribution (is the source reliable?)\n"
            f"- Quantitative claims (are numbers accurate?)\n"
            f"- Causal claims (are cause-effect relationships valid?)"
        )

        return await self._call_for_questions(
            prompt, self.config.batch_model,
            max_q=self.config.max_cove_questions_batch,
        )

    async def _call_for_questions(
        self, prompt: str, model: str, max_q: int,
    ) -> list[VerificationQuestion]:
        """Call LLM for verification questions using structured output.

        Tries structured output first, falls back to text parsing.
        """
        schema_fmt = {
            "type": "json_schema",
            "schema": QUESTIONS_SCHEMA,
        }

        try:
            response = await self.llm_callback(
                prompt, model, output_format=schema_fmt,
            )
        except TypeError:
            # Callback doesn't support output_format kwarg -- fall back
            response = await self.llm_callback(prompt, model)

        # If structured output returned a dict directly, use it
        if isinstance(response, dict):
            items = response.get("questions", [])
            return [
                VerificationQuestion(
                    question=q["question"],
                    aspect=q.get("aspect", "factual"),
                )
                for q in items
                if isinstance(q, dict) and q.get("question")
            ][:max_q]

        # Guard against None/empty response
        if not response:
            return []

        # Fallback: parse text response
        return self._parse_questions(response)[:max_q]

    async def _answer_questions_parallel(
        self,
        questions: list[VerificationQuestion],
        original_finding: str,
        use_batch_model: bool = False,
    ) -> list[VerificationQuestion]:
        """Answer verification questions independently in parallel.

        Two-step CoVe process per question:
        1. Answer the question based on general knowledge (independent)
        2. Compare the answer against the original claim to determine support
        """
        model = (
            self.config.batch_model
            if use_batch_model
            else self.config.streaming_model
        )
        schema_fmt = {"type": "json_schema", "schema": ANSWER_SCHEMA}

        async def answer_question(
            q: VerificationQuestion,
        ) -> VerificationQuestion:
            # Include the original claim so the LLM can actually
            # determine whether its answer supports or contradicts it.
            # The CoVe "factored" variant includes the claim for comparison.
            prompt = (
                "Answer this verification question based on your "
                "knowledge, then determine whether your answer supports "
                "or contradicts the original claim.\n\n"
                f"ORIGINAL CLAIM: {original_finding}\n\n"
                f"VERIFICATION QUESTION: {q.question}\n"
                f"ASPECT: {q.aspect}\n\n"
                "Respond with:\n"
                '- "answer": your factual answer to the question\n'
                '- "confidence": 0.0-1.0 how confident you are\n'
                '- "supports": true if your answer is consistent with '
                "the claim, false if it contradicts it"
            )

            try:
                try:
                    response = await self.llm_callback(
                        prompt, model, output_format=schema_fmt,
                    )
                except TypeError:
                    response = await self.llm_callback(prompt, model)

                if isinstance(response, dict):
                    data = response
                elif response:
                    data = self._parse_json(response)
                else:
                    data = None

                if data:
                    q.independent_answer = data.get("answer", "")
                    q.confidence = data.get("confidence", 0.5)
                    q.supports_original = data.get("supports", True)
            except Exception:
                q.independent_answer = None
                q.confidence = 0.0
                q.supports_original = None

            return q

        # Run all questions in parallel
        answered = await asyncio.gather(
            *[answer_question(q) for q in questions]
        )
        return list(answered)

    def _calculate_consistency(
        self, questions: list[VerificationQuestion]
    ) -> float:
        """Calculate consistency score from answered questions.

        Returns 0-1 score based on:
        - How many questions explicitly support vs contradict
        - Average confidence of answers
        - Questions with unknown support (None) are treated as neutral
        """
        if not questions:
            return 0.5  # No data = neutral, not penalizing

        supporting = 0
        contradicting = 0
        total_confidence = 0.0
        answered_count = 0

        for q in questions:
            if q.independent_answer is not None:
                answered_count += 1
                total_confidence += q.confidence
                # Only count explicit True/False, skip None (unknown)
                if q.supports_original is True:
                    supporting += 1
                elif q.supports_original is False:
                    contradicting += 1

        if answered_count == 0:
            return 0.5  # No data = neutral

        # Support ratio: supporting vs contradicting (ignore unknowns)
        decided_count = supporting + contradicting
        if decided_count > 0:
            support_ratio = supporting / decided_count
        else:
            # All answers have unknown support → neutral
            support_ratio = 0.5

        avg_confidence = total_confidence / answered_count

        # Consistency = (support_ratio * 0.6) + (avg_confidence * 0.4)
        return (support_ratio * 0.6) + (avg_confidence * 0.4)

    def _parse_questions(self, response: str) -> list[VerificationQuestion]:
        """Parse verification questions from LLM response."""
        questions = []
        data = self._extract_json_array(response)
        if data:
            for item in data:
                if isinstance(item, dict) and item.get("question"):
                    questions.append(VerificationQuestion(
                        question=item["question"],
                        aspect=item.get("aspect", "factual"),
                    ))

        # Limit to configured max
        max_questions = self.config.max_cove_questions_batch
        return questions[:max_questions]

    def _extract_json_array(self, response: str) -> list | None:
        """Extract a JSON array from LLM response with multiple fallbacks."""
        if not response or not isinstance(response, str):
            return None

        # Strip markdown code blocks
        cleaned = re.sub(r'```(?:json)?\s*', '', response).strip()
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()

        # Strategy 1: Try parsing the whole cleaned response as JSON
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Greedy regex - match from first [ to last ]
        match = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find balanced brackets
        start = cleaned.find('[')
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == '[':
                    depth += 1
                elif cleaned[i] == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start:i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    def _parse_json(self, response: str) -> dict | None:
        """Parse JSON object from LLM response with multiple fallbacks."""
        if not response or not isinstance(response, str):
            return None

        # Strip markdown code blocks
        cleaned = re.sub(r'```(?:json)?\s*', '', response).strip()
        cleaned = re.sub(r'```\s*$', '', cleaned).strip()

        # Strategy 1: Try parsing the whole cleaned response
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Greedy regex - match from first { to last }
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find balanced braces
        start = cleaned.find('{')
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start:i + 1])
                        except json.JSONDecodeError:
                            break

        return None
