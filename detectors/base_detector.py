import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class Verdict(Enum):
    CLEAN      = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    STEGO      = "STEGO"


class Reliability(Enum):
    HIGH       = "HIGH"
    MEDIUM     = "MEDIUM"
    LOW        = "LOW"
    UNRELIABLE = "UNRELIABLE"


@dataclass
class DetectorResult:
    """
    Standardized output for every detector in the pipeline.

    Attributes:
        probability  : float 0–1. How likely the image contains steganography.
        confidence   : float 0–1. How much to trust this detector's result
                       given the current image type. Overridden by the
                       type-aware layer in Phase 4.
        verdict      : Verdict enum — CLEAN, SUSPICIOUS, or STEGO.
        reliability  : Reliability enum — set based on whether this detector
                       is appropriate for the detected image type.
        detector     : name of the detector that produced this result.
        notes        : human-readable explanation of the result.
        raw_stats    : optional dict of intermediate values for auditability
                       (e.g. chi-square statistic, p-value, RS estimates).
    """
    probability : float
    confidence  : float
    verdict     : Verdict
    reliability : Reliability
    detector    : str
    notes       : str
    raw_stats   : dict = field(default_factory=dict)

    def __post_init__(self):
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(
                f"probability must be in [0, 1], got {self.probability}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )

    def __str__(self):
        return (
            f"[{self.detector}] "
            f"Verdict: {self.verdict.value} | "
            f"Probability: {self.probability:.3f} | "
            f"Confidence: {self.confidence:.3f} | "
            f"Reliability: {self.reliability.value}\n"
            f"  Notes: {self.notes}"
        )


def probability_to_verdict(probability: float) -> Verdict:
    """
    Convert a probability score to a Verdict enum.
    Thresholds:
        < 0.40  → CLEAN
        0.40 – 0.69 → SUSPICIOUS
        >= 0.70 → STEGO
    """
    if probability >= 0.70:
        return Verdict.STEGO
    elif probability >= 0.40:
        return Verdict.SUSPICIOUS
    else:
        return Verdict.CLEAN


class BaseDetector(ABC):
    """
    Abstract base class for all steganography detectors.
    Every detector must implement the analyze() method and return
    a DetectorResult. No detector should raise an unhandled exception —
    all failure modes must be caught and returned as a result with
    low confidence and an explanatory note.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name."""
        ...

    @abstractmethod
    def analyze(self, image: "np.ndarray") -> DetectorResult:
        """
        Analyze an image array for steganographic content.

        Args:
            image: RGB image as a NumPy array (H x W x 3, uint8)

        Returns:
            DetectorResult with all fields populated.
        """
        ...