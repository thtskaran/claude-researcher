"""Confidence calibration for verification results."""

from dataclasses import dataclass
from typing import Optional

from .models import VerificationConfig, VerificationStatus


@dataclass
class CalibrationResult:
    """Result of confidence calibration."""
    original_confidence: float
    calibrated_confidence: float
    status: VerificationStatus
    adjustments: list[tuple[str, float]]  # (reason, delta)


class ConfidenceCalibrator:
    """Calibrates confidence scores and determines verification status.

    Based on research thresholds:
    - >85%: Auto-accept (VERIFIED)
    - 50-85%: Flag for review (FLAGGED)
    - <50%: Reject (REJECTED)
    """

    def __init__(self, config: VerificationConfig | None = None):
        self.config = config or VerificationConfig()

    def calibrate(
        self,
        original_confidence: float,
        cove_consistency_score: float = 0.0,
        kg_support_score: float = 0.0,
        has_contradictions: bool = False,
        source_credibility: float = 0.0,
        critic_corrections: int = 0,
    ) -> CalibrationResult:
        """Calibrate confidence based on verification signals.

        Args:
            original_confidence: The original finding confidence (0-1)
            cove_consistency_score: How consistent CoVe answers are (0-1)
            kg_support_score: KG corroboration score (0-1)
            has_contradictions: Whether contradictions were detected
            source_credibility: Source credibility score (0-1)
            critic_corrections: Number of CRITIC corrections made

        Returns:
            CalibrationResult with calibrated confidence and status
        """
        adjustments = []
        calibrated = original_confidence

        # CoVe consistency adjustment
        if cove_consistency_score > 0:
            # High consistency boosts confidence, low consistency reduces it
            cove_delta = (cove_consistency_score - 0.5) * 0.2  # +/-10% max
            calibrated += cove_delta
            if abs(cove_delta) > 0.01:
                adjustments.append(("cove_consistency", cove_delta))

        # KG support adjustment
        if kg_support_score > 0:
            kg_delta = kg_support_score * self.config.kg_support_boost
            calibrated += kg_delta
            if kg_delta > 0.01:
                adjustments.append(("kg_support", kg_delta))

        # Contradiction penalty
        if has_contradictions:
            calibrated -= self.config.kg_contradiction_penalty
            adjustments.append(("contradiction_penalty", -self.config.kg_contradiction_penalty))

        # Source credibility adjustment
        if source_credibility > 0:
            # Credibility above 0.7 boosts, below reduces
            cred_delta = (source_credibility - 0.7) * 0.15  # +/-4.5% max
            calibrated += cred_delta
            if abs(cred_delta) > 0.01:
                adjustments.append(("source_credibility", cred_delta))

        # CRITIC correction penalty
        if critic_corrections > 0:
            # Each correction reduces confidence slightly (finding needed correction)
            correction_penalty = min(critic_corrections * 0.05, 0.15)  # Max 15% penalty
            calibrated -= correction_penalty
            adjustments.append(("critic_corrections", -correction_penalty))

        # Clamp to valid range
        calibrated = max(0.0, min(1.0, calibrated))

        # Determine status based on thresholds
        status = self._determine_status(calibrated)

        return CalibrationResult(
            original_confidence=original_confidence,
            calibrated_confidence=calibrated,
            status=status,
            adjustments=adjustments,
        )

    def _determine_status(self, confidence: float) -> VerificationStatus:
        """Determine verification status from calibrated confidence."""
        if confidence >= self.config.auto_accept_threshold:
            return VerificationStatus.VERIFIED
        elif confidence >= self.config.flag_threshold:
            return VerificationStatus.FLAGGED
        else:
            return VerificationStatus.REJECTED

    def should_use_critic(self, confidence: float) -> bool:
        """Determine if CRITIC should be used for deeper verification."""
        return (
            self.config.enable_critic
            and confidence < self.config.critic_confidence_threshold
        )

    def calculate_batch_summary(
        self,
        results: list[CalibrationResult],
    ) -> dict:
        """Calculate summary statistics for a batch of calibrations."""
        if not results:
            return {
                "total": 0,
                "verified": 0,
                "flagged": 0,
                "rejected": 0,
                "avg_original": 0.0,
                "avg_calibrated": 0.0,
                "avg_adjustment": 0.0,
            }

        verified = sum(1 for r in results if r.status == VerificationStatus.VERIFIED)
        flagged = sum(1 for r in results if r.status == VerificationStatus.FLAGGED)
        rejected = sum(1 for r in results if r.status == VerificationStatus.REJECTED)

        avg_original = sum(r.original_confidence for r in results) / len(results)
        avg_calibrated = sum(r.calibrated_confidence for r in results) / len(results)
        avg_adjustment = avg_calibrated - avg_original

        return {
            "total": len(results),
            "verified": verified,
            "flagged": flagged,
            "rejected": rejected,
            "verified_pct": verified / len(results) * 100,
            "flagged_pct": flagged / len(results) * 100,
            "rejected_pct": rejected / len(results) * 100,
            "avg_original": avg_original,
            "avg_calibrated": avg_calibrated,
            "avg_adjustment": avg_adjustment,
        }
