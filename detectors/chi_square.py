"""
detectors/chi_square.py — Chi-Square LSB pair equality detector

Purpose:
    Detects LSB replacement steganography by testing whether adjacent
    pixel value pairs (0,1), (2,3), (4,5)... are suspiciously equal.

How it works:
    In a natural image, pair counts are uneven: value 128 might appear
    5000 times and value 129 might appear 3200 times. When LSB replacement
    is applied, flipping an LSB converts 128 -> 129 or vice versa,
    forcing these pairs toward equality. The chi-square test measures
    how statistically unlikely this equality is in a natural image.

Critical probability direction:
    probability = p_value directly.
    A HIGH p_value means the test FAILED to reject pair equality.
    Failing to reject equality IS the stego signal.
    NEVER use 1 - p_value here — that would invert the detection.

Interview relevance: HIGH (classic steganography detection method)
"""

import numpy as np
from scipy.stats import chisquare
from detectors.base_detector import (
    BaseDetector,
    DetectorResult,
    Reliability,
    probability_to_verdict,
)


class ChiSquareDetector(BaseDetector):
    """
    Detects LSB steganography by measuring pixel value pair equality.

    Known limitations (important for interviews):
        - Unreliable on scanned/analog images (Mandrill problem)
        - Unreliable on AI-generated images
        - Fails on payloads below ~3% of capacity
        - Only analyzes the R channel by default (standard practice)
    """

    @property
    def name(self) -> str:
        return "Chi-Square"

    def analyze(self, image: np.ndarray) -> DetectorResult:
        """
        Run chi-square analysis on the R channel of the image.

        Args:
            image: RGB image array (H x W x 3, uint8)

        Returns:
            DetectorResult with probability = p_value from chi-square test.
        """
        try:
            r_channel = image[:, :, 0].flatten()
            counts = np.bincount(r_channel, minlength=256)

            # Build observed and expected pair arrays.
            # If LSB replacement occurred, each pair's total should be
            # split equally between the two values.
            observed  = []
            expected  = []

            for i in range(0, 256, 2):
                pair_total = counts[i] + counts[i + 1]
                if pair_total == 0:
                    continue
                observed.append(counts[i])
                observed.append(counts[i + 1])
                expected.append(pair_total / 2)
                expected.append(pair_total / 2)

            observed = np.array(observed, dtype=np.float64)
            expected = np.array(expected, dtype=np.float64)

            mask = expected > 0
            observed = observed[mask]
            expected = expected[mask]

            if len(observed) < 2:
                return DetectorResult(
                    probability=0.0,
                    confidence=0.1,
                    verdict=probability_to_verdict(0.0),
                    reliability=Reliability.LOW,
                    detector=self.name,
                    notes="Insufficient pixel value diversity for chi-square analysis.",
                    raw_stats={},
                )

            _, p_value = chisquare(f_obs=observed, f_exp=expected)

            probability = float(p_value)

            notes = (
                f"p-value: {p_value:.4f}. "
                f"{'High p-value suggests pair equality consistent with LSB embedding.' if p_value > 0.4 else 'Low p-value suggests natural unequal pair distribution.'} "
                f"Note: unreliable on scanned and AI-generated images."
            )

            return DetectorResult(
                probability=probability,
                confidence=0.75,
                verdict=probability_to_verdict(probability),
                reliability=Reliability.HIGH,
                detector=self.name,
                notes=notes,
                raw_stats={"p_value": p_value},
            )

        except Exception as e:
            return DetectorResult(
                probability=0.0,
                confidence=0.0,
                verdict=probability_to_verdict(0.0),
                reliability=Reliability.UNRELIABLE,
                detector=self.name,
                notes=f"Chi-square analysis failed: {str(e)}",
                raw_stats={},
            )