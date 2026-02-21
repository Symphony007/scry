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

    How it works:
        Divides the image into small pixel groups. Applies a flipping mask
        to each group and measures the smoothness (variation) of the group
        before and after flipping. Groups are classified as:
            Regular (R)  : flipping increases variation
            Singular (S) : flipping decreases variation
            Unusable (U) : no change

        This is done with both a normal mask (F) and a negative mask (-F).
        In a clean image: R_m ≈ R_{-m} and S_m ≈ S_{-m}
        After LSB embedding: R_m > R_{-m} and S_m < S_{-m}

        The payload size is estimated using the RS quadratic formula derived
        from the asymmetry between normal and negative mask responses.

    Why it is the most powerful single detector:
        Unlike chi-square (which only looks at pair counts) or entropy
        (which only looks at randomness), RS analysis captures the spatial
        smoothness disruption caused by embedding — a fundamentally different
        signal that is harder to fake.

    Known limitations:
        - Requires smooth spatial regions to work — fails on pure noise arrays
        - Affected by the Mandrill's film grain (reduced reliability on scanned)
        - Computationally heavier than other detectors
        - Fails on payloads below ~3% of capacity
    """

    GROUP_SIZE = 8  # pixels per group (1 row of an 8-pixel horizontal block)

    # Flipping mask F: standard RS analysis mask
    MASK = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int32)

    @property
    def name(self) -> str:
        return "RS Analysis"

    def _flip(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply normal bit-flip to values where mask == 1.
        Normal flip: 0↔1, 2↔3, 4↔5, ... (LSB flip)
        """
        result = values.copy().astype(np.int32)
        for i, m in enumerate(mask):
            if m == 1:
                # Flip LSB: even→odd, odd→even
                result[i] = result[i] ^ 1
        return result

    def _negative_flip(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply negative bit-flip to values where mask == 1.
        Negative flip: 1↔0, 3↔2, 5↔4, ... (inverse LSB flip)
        Maps x → x-1 if odd, x+1 if even (within bounds).
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
        Measure the smoothness of a pixel group as the sum of absolute
        differences between adjacent pixels. Lower = smoother.
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

        Solves: 2(d1 + d0)x^2 - (2d0 + d1)x + d0 = 0
        where d0 = r_m - s_m, d1 = rm - sm (normalized differences)

        Returns estimated payload as a fraction [0, 1], or 0.0 if
        the quadratic has no valid solution.
        """
        d0 = float(r_m - s_m)
        d1 = float(rm  - sm)

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

        # Choose the root in [0, 0.5] — payload fraction is bounded
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
            channel = image[:, :, 0]  # R channel

            # Normal mask classifications
            Rm, Sm, _ = self._classify_groups(channel, self.MASK, negative=False)
            # Negative mask classifications
            R_m, S_m, _ = self._classify_groups(channel, self.MASK, negative=True)

            total = max(Rm + Sm, 1)  # avoid division by zero

            # Normalize
            rm  = Rm  / total
            sm  = Sm  / total
            r_m = R_m / total
            s_m = S_m / total

            # Asymmetry score: the core RS signal
            # Clean image  → asymmetry ≈ 0
            # Stego image  → rm > r_m and sm < s_m → asymmetry > 0
            asymmetry = float((rm - r_m) - (sm - s_m))

            # Estimate payload
            payload_estimate = self._estimate_payload(rm, sm, r_m, s_m)

            # Map asymmetry to probability
            # Asymmetry of 0.05+ is a meaningful signal
            # Asymmetry of 0.20+ is a strong signal
            LOW  = 0.02
            HIGH = 0.20

            if asymmetry <= LOW:
                probability = 0.0
            elif asymmetry >= HIGH:
                probability = 1.0
            else:
                probability = (asymmetry - LOW) / (HIGH - LOW)

            probability = float(np.clip(probability, 0.0, 1.0))

            notes = (
                f"RS asymmetry: {asymmetry:.4f}. "
                f"Rm={rm:.3f}, Sm={sm:.3f}, R-m={r_m:.3f}, S-m={s_m:.3f}. "
                f"Estimated payload: {payload_estimate * 100:.1f}% of capacity. "
                f"{'Strong asymmetry — consistent with LSB embedding.' if asymmetry > 0.1 else 'Low asymmetry — consistent with clean image.'}"
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