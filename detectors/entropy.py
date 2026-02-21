import numpy as np
from detectors.base_detector import (
    BaseDetector,
    DetectorResult,
    Reliability,
    probability_to_verdict,
)


class EntropyDetector(BaseDetector):
    """
    Detects LSB steganography by measuring LSB plane randomness.

    How it works:
        In a natural image, the LSB plane has structure — it is not purely
        random. When a message is embedded, the LSBs in the embedded region
        are replaced with message bits, which are effectively random. This
        raises LSB entropy toward 1.0 in the embedded region.

    Probability interpretation:
        High entropy → high probability of embedding.
        Entropy is measured per 8x8 block and averaged across the image.
        The score is normalized to [0, 1].

    Known limitations (CRITICAL — this detector is a supporting signal only):
        - USELESS on AI-generated images: diffusion models produce near-perfect
          LSB entropy by construction. High entropy is baseline, not a signal.
        - USELESS on high-ISO photographs: sensor noise randomizes LSBs.
        - USELESS on scanned images: film grain randomizes LSBs.
        - Fails on payloads below ~3% of capacity.
        - Should never be the primary or sole detection signal.
        - Weight is set low (0.5) in the aggregator by design.
    """

    BLOCK_SIZE = 8

    @property
    def name(self) -> str:
        return "Entropy"

    def _block_entropy(self, bits: np.ndarray) -> float:
        """
        Compute Shannon entropy of a 1D binary array.
        Returns a value in [0, 1] where 1.0 is maximum randomness.
        """
        if len(bits) == 0:
            return 0.0

        p1 = np.mean(bits)
        p0 = 1.0 - p1

        if p1 == 0.0 or p1 == 1.0:
            return 0.0

        entropy = -(p0 * np.log2(p0) + p1 * np.log2(p1))
        return float(entropy)  # already in [0, 1] for binary

    def analyze(self, image: np.ndarray) -> DetectorResult:
        """
        Measure LSB entropy across the full image using block averaging.

        Args:
            image: RGB image array (H x W x 3, uint8)

        Returns:
            DetectorResult with probability derived from mean LSB entropy.
        """
        try:
            h, w, _ = image.shape

            # Extract LSB plane from all three channels combined
            lsb_plane = (image & 1).astype(np.uint8)
            lsb_flat  = lsb_plane.flatten()

            # Compute entropy over fixed-size blocks and average
            block_entropies = []
            block_pixels = self.BLOCK_SIZE * self.BLOCK_SIZE * 3

            for i in range(0, len(lsb_flat) - block_pixels, block_pixels):
                block = lsb_flat[i: i + block_pixels]
                block_entropies.append(self._block_entropy(block))

            if not block_entropies:
                # Image too small for block analysis — use full plane
                mean_entropy = self._block_entropy(lsb_flat)
            else:
                mean_entropy = float(np.mean(block_entropies))

            # Normalize: natural images cluster around 0.7–0.85 entropy.
            # We map entropy to probability such that:
            #   entropy <= 0.7  → low probability (natural structure)
            #   entropy >= 0.95 → high probability (near-random = stego signal)
            LOW_THRESHOLD  = 0.70
            HIGH_THRESHOLD = 0.95

            if mean_entropy <= LOW_THRESHOLD:
                probability = 0.0
            elif mean_entropy >= HIGH_THRESHOLD:
                probability = 1.0
            else:
                probability = (mean_entropy - LOW_THRESHOLD) / (
                    HIGH_THRESHOLD - LOW_THRESHOLD
                )

            probability = float(np.clip(probability, 0.0, 1.0))

            notes = (
                f"Mean LSB entropy: {mean_entropy:.4f}. "
                f"{'High entropy suggests possible embedding.' if probability > 0.4 else 'Entropy consistent with natural image structure.'} "
                f"WARNING: This detector is unreliable on AI-generated images, "
                f"high-ISO photos, and scanned images where LSB entropy is "
                f"naturally high. Use as supporting signal only."
            )

            return DetectorResult(
                probability=probability,
                confidence=0.5,  # intentionally low — supporting signal only
                verdict=probability_to_verdict(probability),
                reliability=Reliability.MEDIUM,
                detector=self.name,
                notes=notes,
                raw_stats={
                    "mean_entropy": mean_entropy,
                    "block_count": len(block_entropies),
                },
            )

        except Exception as e:
            return DetectorResult(
                probability=0.0,
                confidence=0.0,
                verdict=probability_to_verdict(0.0),
                reliability=Reliability.UNRELIABLE,
                detector=self.name,
                notes=f"Entropy analysis failed: {str(e)}",
                raw_stats={},
            )