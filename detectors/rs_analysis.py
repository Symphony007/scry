"""
detectors/rs_analysis.py — Regular-Singular (RS) Analysis detector

Purpose:
    Detects LSB steganography by measuring the spatial smoothness disruption
    that embedding causes in pixel groups. Also estimates how much of the
    image capacity has been used (the payload estimate).

How it works:
    Groups of 8 pixels are classified as Regular (R), Singular (S),
    or Unusable (U) by applying a flipping mask and measuring how
    the group's smoothness (variation) changes:
        Regular  : flipping increases variation
        Singular : flipping decreases variation

    This is done with both a normal mask (F) and a negative mask (-F).
    In a clean image: R_m ≈ R_{-m} and S_m ≈ S_{-m}
    After LSB embedding: R_m > R_{-m} and S_m < S_{-m}

    The asymmetry between normal/negative mask responses is the detection signal.
    The payload size is estimated from the asymmetry using a quadratic formula.

Why it is the strongest single detector:
    Chi-square only looks at pair counts in the histogram.
    RS analysis looks at spatial smoothness patterns — a fundamentally
    different signal that is much harder to fake or erase.

Interview relevance: HIGH (most reliable detector in the pipeline)
"""

import numpy as np
from detectors.base_detector import (
    BaseDetector,
    DetectorResult,
    Reliability,
    probability_to_verdict,
)


class RSAnalysisDetector(BaseDetector):
    """
    Detects LSB steganography using Regular-Singular (RS) Analysis.

    Known limitations:
        - Requires smooth spatial regions — fails on pure noise
        - Affected by heavy film grain (reduced reliability on scanned images)
        - Computationally heavier than other detectors
        - Fails on payloads below ~3% of capacity
    """

    GROUP_SIZE = 8

    # Standard RS analysis mask — selects which pixels in a group to flip.
    # This specific mask is defined in the original Fridrich et al. paper.
    MASK = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)

    @property
    def name(self) -> str:
        return "RS Analysis"

    def _flip(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Normal LSB flip: XOR with 1 at mask positions.
        Even values become odd, odd values become even.
        """
        result = values.copy().astype(np.int32)
        for i, m in enumerate(mask):
            if m == 1:
                result[i] = result[i] ^ 1
        return result

    def _negative_flip(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Negative flip (inverse operation): at mask positions,
        even values decrease by 1, odd values increase by 1.
        This is the mathematical inverse of the normal flip.
        """
        result = values.copy().astype(np.int32)
        for i, m in enumerate(mask):
            if m == 1:
                if result[i] % 2 == 0:
                    result[i] = max(0, result[i] - 1)
                else:
                    result[i] = min(255, result[i] + 1)
        return result

    def _smoothness(self, values: np.ndarray) -> float:
        """
        Sum of absolute differences between adjacent pixels in a group.
        Lower = smoother group. This is the discriminating function f().
        """
        return float(np.sum(np.abs(np.diff(values.astype(np.float64)))))

    def _classify_groups(
        self, channel: np.ndarray, mask: np.ndarray, negative: bool = False
    ) -> tuple[int, int, int]:
        """
        Classify all pixel groups in a channel as Regular, Singular, or Unusable.

        Args:
            channel : 2D array (H x W) for one color channel
            mask    : flipping mask array
            negative: if True, use negative flip instead of normal flip

        Returns:
            (R_count, S_count, U_count)
        """
        h, w = channel.shape
        R = S = U = 0
        group_size = len(mask)

        for row in range(h):
            for col in range(0, w - group_size + 1, group_size):
                group = channel[row, col: col + group_size].astype(np.int32)

                if negative:
                    flipped = self._negative_flip(group, mask)
                else:
                    flipped = self._flip(group, mask)

                f_original = self._smoothness(group)
                f_flipped  = self._smoothness(flipped)

                if f_flipped > f_original:
                    R += 1
                elif f_flipped < f_original:
                    S += 1
                else:
                    U += 1

        return R, S, U

    def _estimate_payload(
        self, rm: float, sm: float, r_m: float, s_m: float
    ) -> float:
        """
        Estimate payload fraction using the RS quadratic formula.

        The formula is derived from the relationship between normal and
        negative mask R/S counts as a function of embedding rate.
        Solves: 2(d1 + d0)x^2 - (2d0 + d1)x + d0 = 0

        Returns estimated payload as a fraction [0, 0.5], or 0.0 if
        the quadratic has no valid solution.
        """
        d0 = r_m - s_m
        d1 = rm  - sm

        a = 2 * (d1 + d0)
        b = -(2 * d0 + d1)
        c = d0

        if abs(a) < 1e-10:
            return 0.0

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return 0.0

        x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        x2 = (-b - np.sqrt(discriminant)) / (2 * a)

        # Payload fraction is bounded between 0 and 0.5
        candidates = [x for x in [x1, x2] if 0.0 <= x <= 0.5]
        if not candidates:
            return 0.0

        return float(min(candidates))

    def analyze(self, image: np.ndarray) -> DetectorResult:
        """
        Run RS analysis on the image's R channel.

        Args:
            image: RGB image array (H x W x 3, uint8)

        Returns:
            DetectorResult with probability derived from RS asymmetry
            and an estimated payload percentage.
        """
        try:
            channel = image[:, :, 0]

            Rm, Sm, _ = self._classify_groups(channel, self.MASK, negative=False)
            R_m, S_m, _ = self._classify_groups(channel, self.MASK, negative=True)

            total = max(Rm + Sm, 1)

            rm  = Rm  / total
            sm  = Sm  / total
            r_m = R_m / total
            s_m = S_m / total

            # Asymmetry is the core RS signal.
            # Clean image  -> asymmetry ~= 0
            # Stego image  -> rm > r_m and sm < s_m -> asymmetry > 0
            asymmetry = (rm - r_m) - (sm - s_m)

            payload_estimate = self._estimate_payload(rm, sm, r_m, s_m)

            # Map asymmetry to probability using linear interpolation.
            # Asymmetry of 0.05+ is a meaningful signal; 0.20+ is strong.
            LOW  = 0.02
            HIGH = 0.20

            if asymmetry <= LOW:
                probability = 0.0
            elif asymmetry >= HIGH:
                probability = 1.0
            else:
                probability = (asymmetry - LOW) / (HIGH - LOW)

            probability = float(np.clip(probability, 0.0, 1.0))

            asymmetry_note = (
                "Strong asymmetry — consistent with LSB embedding."
                if asymmetry > 0.1
                else "Low asymmetry — consistent with clean image."
            )
            notes = (
                f"RS asymmetry: {asymmetry:.4f}. "
                f"Rm={rm:.3f}, Sm={sm:.3f}, R-m={r_m:.3f}, S-m={s_m:.3f}. "
                f"Estimated payload: {payload_estimate * 100:.1f}% of capacity. "
                f"{asymmetry_note}"
            )

            return DetectorResult(
                probability=probability,
                confidence=0.85,
                verdict=probability_to_verdict(probability),
                reliability=Reliability.HIGH,
                detector=self.name,
                notes=notes,
                raw_stats={
                    "asymmetry": asymmetry,
                    "rm": rm, "sm": sm,
                    "r_m": r_m, "s_m": s_m,
                    "payload_estimate_pct": payload_estimate * 100,
                },
            )

        except Exception as e:
            return DetectorResult(
                probability=0.0,
                confidence=0.0,
                verdict=probability_to_verdict(0.0),
                reliability=Reliability.UNRELIABLE,
                detector=self.name,
                notes=f"RS analysis failed: {str(e)}",
                raw_stats={},
            )