"""Verification system for fact checking and hallucination reduction.

This module implements Chain-of-Verification (CoVe) + CRITIC to achieve
50-70% hallucination reduction through:

1. Streaming verification - Real-time verification when interns submit findings (<500ms)
2. Batch verification - Comprehensive verification during manager synthesis (~2s/finding)

Confidence Thresholds:
- >85%: Auto-accept (VERIFIED)
- 50-85%: Flag for review (FLAGGED)
- <50%: Reject, trigger additional search (REJECTED)
"""

from .models import (
    VerificationStatus,
    VerificationMethod,
    VerificationQuestion,
    VerificationResult,
    BatchVerificationResult,
    VerificationConfig,
    ContradictionDetail,
)

from .confidence import (
    ConfidenceCalibrator,
    CalibrationResult,
)

from .cove import ChainOfVerification

from .critic import (
    CRITICVerifier,
    HighStakesDetector,
)

from .metrics import (
    VerificationMetricsTracker,
    LatencyMetrics,
)

from .pipeline import (
    VerificationPipeline,
    create_verification_pipeline,
)

__all__ = [
    # Models
    "VerificationStatus",
    "VerificationMethod",
    "VerificationQuestion",
    "VerificationResult",
    "BatchVerificationResult",
    "VerificationConfig",
    "ContradictionDetail",

    # Confidence
    "ConfidenceCalibrator",
    "CalibrationResult",

    # CoVe
    "ChainOfVerification",

    # CRITIC
    "CRITICVerifier",
    "HighStakesDetector",

    # Metrics
    "VerificationMetricsTracker",
    "LatencyMetrics",

    # Pipeline
    "VerificationPipeline",
    "create_verification_pipeline",
]
