# detectors/histogram.py

import numpy as np
from detectors.base_detector import (
    BaseDetector,
    DetectorResult,
    Reliability,
    probability_to_verdict,
)


class HistogramDetector(BaseDetector):
    """
    Detects LSB steganography by measuring histogram combing.

    How it works:
        LSB replacement converts pixel values to their LSB-flipped neighbor:
        128 → 129 or 129 → 128 depending on the message bit. Over many pixels
        this causes adjacent value pairs (0,1), (2,3), (4,5)... to become
        suspiciously equal in count — producing a sawtooth "combing" pattern
        in the histogram.

        The combing score measures how equal adjacent pairs are across the
        full 0–255 value range. A perfectly combed histogram scores 1.0.
        A natural histogram scores close to 0.0.

    Known limitations:
        - Unreliable on smooth gradient images where adjacent pair counts
          are naturally similar regardless of embedding
        - Unreliable on scanned images
        - Fails on payloads below ~3% of capacity
        - Works best on images with a narrow or skewed pixel distribution
    """

    @property
    def name(self) -> str:
        return "Histogram"

    def _combing_score(self, channel: np.ndarray) -> float:
        """
        Compute the histogram combing score for a single channel.

        For each adjacent pair (2i, 2i+1), measure how equal their counts
        are. Perfect equality → score 1.0. Maximum inequality → score 0.0.

        Returns a float in [0, 1].
        """
        counts = np.bincount(channel.flatten(), minlength=256).astype(np.float64)

        pair_scores = []
        for i in range(0, 256, 2):
            a = counts[i]
            b = counts[i + 1]
            total = a + b
            if total == 0:
                continue
            # Equality score: 1.0 when perfectly equal, 0.0 when all in one
            equality = 1.0 - abs(a - b) / total
            pair_scores.append(equality)

        if not pair_scores:
            return 0.0

        return float(np.mean(pair_scores))

    def analyze(self, image: np.ndarray) -> DetectorResult:
        """
        Measure histogram combing across all three RGB channels.

        Args:
            image: RGB image array (H x W x 3, uint8)

        Returns:
            DetectorResult with probability derived from mean combing score.
        """
        try:
            scores = []
            channel_names = ["R", "G", "B"]
            channel_scores = {}

            for i, name in enumerate(channel_names):
                score = self._combing_score(image[:, :, i])
                scores.append(score)
                channel_scores[name] = score

            mean_score = float(np.mean(scores))

            # Natural images cluster around 0.6–0.75 combing score
            # because adjacent values are somewhat similar by nature.
            # Embedded images push toward 0.90+.
            LOW_THRESHOLD  = 0.70
            HIGH_THRESHOLD = 0.92

            if mean_score <= LOW_THRESHOLD:
                probability = 0.0
            elif mean_score >= HIGH_THRESHOLD:
                probability = 1.0
            else:
                probability = (mean_score - LOW_THRESHOLD) / (
                    HIGH_THRESHOLD - LOW_THRESHOLD
                )

            probability = float(np.clip(probability, 0.0, 1.0))

            notes = (
                f"Mean combing score: {mean_score:.4f}. "
                f"R={channel_scores['R']:.3f}, "
                f"G={channel_scores['G']:.3f}, "
                f"B={channel_scores['B']:.3f}. "
                f"{'High combing — adjacent pair counts suspiciously equal.' if probability > 0.4 else 'Low combing — consistent with natural pixel distribution.'} "
                f"Note: unreliable on smooth gradients and scanned images."
            )

            return DetectorResult(
                probability=probability,
                confidence=0.70,
                verdict=probability_to_verdict(probability),
                reliability=Reliability.MEDIUM,
                detector=self.name,
                notes=notes,
                raw_stats={
                    "mean_combing_score": mean_score,
                    "channel_scores": channel_scores,
                },
            )

        except Exception as e:
            return DetectorResult(
                probability=0.0,
                confidence=0.0,
                verdict=probability_to_verdict(0.0),
                reliability=Reliability.UNRELIABLE,
                detector=self.name,
                notes=f"Histogram analysis failed: {str(e)}",
                raw_stats={},
            )