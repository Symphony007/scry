"""
detectors/aggregator.py — Weighted score aggregator

Purpose:
    Combines the individual results from all detectors into a single
    final verdict and probability. This is where type-aware weighting happens.

Inputs:
    - A list of DetectorResult objects (one per detector)
    - Optional: a custom weight dict (from the type-aware layer)

Outputs:
    - AggregatorResult with: final probability, verdict, confidence,
      per-detector weights used, and payload estimate from RS Analysis.

Why weighted aggregation:
    Not all detectors are equally reliable on all image types.
    On a scanned photo (Mandrill problem), chi-square gives false positives
    because film grain naturally creates pair equality. The type-aware
    layer reduces chi-square's weight to 0.3 for scanned images.
    This is the key architectural insight of the project.

How it fits in:
    build_type_aware_aggregator() is called after image type classification.
    It reads the weight table from ml/type_classifier.py's WEIGHT_TABLES
    and returns a configured ScoreAggregator ready to combine results.
"""

from __future__ import annotations

import numpy as np
from detectors.base_detector import DetectorResult, Verdict, Reliability, probability_to_verdict


# Default weights used when no type-specific table is available.
# RS Analysis has the highest weight because it captures spatial disruption —
# a signal that is harder to remove than the histogram fingerprint chi-square sees.
DEFAULT_WEIGHTS = {
    "RS Analysis" : 2.0,
    "Chi-Square"  : 1.5,
    "Histogram"   : 1.0,
    "Entropy"     : 0.5,
}


class AggregatorResult:
    """
    Combined output from the score aggregator.

    Attributes:
        final_probability : weighted average probability across all detectors
        final_verdict     : Verdict enum derived from final_probability
        confidence        : weighted average confidence across all detectors
        detector_results  : list of individual DetectorResult objects
        weights_used      : dict of detector name -> weight actually applied
        payload_estimate  : best payload estimate from RS Analysis if available
        notes             : plain-English summary of the aggregation
    """

    def __init__(
        self,
        final_probability : float,
        final_verdict     : Verdict,
        confidence        : float,
        detector_results  : list[DetectorResult],
        weights_used      : dict[str, float],
        payload_estimate  : float | None,
        notes             : str,
    ):
        self.final_probability = final_probability
        self.final_verdict     = final_verdict
        self.confidence        = confidence
        self.detector_results  = detector_results
        self.weights_used      = weights_used
        self.payload_estimate  = payload_estimate
        self.notes             = notes

    def __str__(self):
        lines = [
            f"OVERALL ASSESSMENT: {self.final_verdict.value} "
            f"({self.final_probability * 100:.1f}% probability)",
            f"Confidence: {self.confidence:.3f}",
            "",
            "DETECTOR BREAKDOWN",
        ]
        for r in self.detector_results:
            weight = self.weights_used.get(r.detector, 0.0)
            lines.append(
                f"  {r.detector:<15} "
                f"prob={r.probability:.3f}  "
                f"weight={weight:.1f}  "
                f"reliability={r.reliability.value}"
            )
        if self.payload_estimate is not None:
            lines.append(f"\nPayload estimate: {self.payload_estimate:.1f}% of capacity")
        lines.append(f"\n{self.notes}")
        return "\n".join(lines)


class ScoreAggregator:
    """
    Combines detector results into a single weighted verdict.

    Design principles:
        - Accepts a weight dictionary so the type-aware layer can override
          weights per image type without modifying the aggregator.
        - Detectors with weight 0.0 are excluded from aggregation entirely.
        - Payload estimate is taken from RS Analysis when available.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        """
        Args:
            weights: dict mapping detector name to float weight.
                     Pass a type-specific dict to override for a specific image type.
                     Defaults to DEFAULT_WEIGHTS if not provided.
        """
        self.weights = weights if weights is not None else DEFAULT_WEIGHTS.copy()

    def aggregate(self, results: list[DetectorResult]) -> AggregatorResult:
        """
        Combine a list of DetectorResult objects into a single AggregatorResult.

        Args:
            results: list of DetectorResult from each detector

        Returns:
            AggregatorResult with weighted probability, verdict, and breakdown.
        """
        if not results:
            return AggregatorResult(
                final_probability = 0.0,
                final_verdict     = Verdict.CLEAN,
                confidence        = 0.0,
                detector_results  = [],
                weights_used      = {},
                payload_estimate  = None,
                notes             = "No detector results provided.",
            )

        weighted_prob_sum  = 0.0
        weighted_conf_sum  = 0.0
        total_weight       = 0.0
        weights_used       = {}
        payload_estimate   = None

        for result in results:
            weight = self.weights.get(result.detector, 0.0)
            weights_used[result.detector] = weight

            if weight == 0.0:
                continue

            weighted_prob_sum += result.probability * weight
            weighted_conf_sum += result.confidence  * weight
            total_weight      += weight

            # Use RS Analysis payload estimate when available —
            # it's the most reliable payload size estimate in the pipeline.
            if result.detector == "RS Analysis":
                rs_payload = result.raw_stats.get("payload_estimate_pct")
                if rs_payload is not None:
                    payload_estimate = float(rs_payload)

        if total_weight == 0.0:
            return AggregatorResult(
                final_probability = 0.0,
                final_verdict     = Verdict.CLEAN,
                confidence        = 0.0,
                detector_results  = results,
                weights_used      = weights_used,
                payload_estimate  = None,
                notes             = "All detector weights are zero \u2014 manual review required.",
            )

        final_probability = weighted_prob_sum / total_weight
        final_confidence  = weighted_conf_sum / total_weight
        final_verdict     = probability_to_verdict(final_probability)

        unreliable = [
            r.detector for r in results
            if r.reliability in (Reliability.LOW, Reliability.UNRELIABLE)
            and self.weights.get(r.detector, 0.0) > 0.0
        ]

        notes_parts = [
            f"Weighted aggregation across "
            f"{len([w for w in weights_used.values() if w > 0])} "
            f"active detectors (total weight: {total_weight:.1f})."
        ]
        if unreliable:
            notes_parts.append(
                f"Low-reliability detectors in active set: "
                f"{', '.join(unreliable)}. "
                f"Interpret results with caution."
            )

        return AggregatorResult(
            final_probability = float(np.clip(final_probability, 0.0, 1.0)),
            final_verdict     = final_verdict,
            confidence        = float(np.clip(final_confidence,  0.0, 1.0)),
            detector_results  = results,
            weights_used      = weights_used,
            payload_estimate  = payload_estimate,
            notes             = " ".join(notes_parts),
        )


def build_type_aware_aggregator(classification_result) -> ScoreAggregator:
    """
    Build a ScoreAggregator with weights tuned for the detected image type.

    This is the critical function that connects image type classification
    (ml/classifier.py) to score aggregation (this file).

    Without this function, every image would use DEFAULT_WEIGHTS —
    causing false positives on scanned and AI-generated images because
    those image types naturally exhibit stego-like statistical properties.

    Args:
        classification_result: ClassificationResult from HierarchicalClassifier

    Returns:
        ScoreAggregator configured with type-appropriate detector weights.
    """
    from ml.type_classifier import WEIGHT_TABLES
    from ml.type_features import ImageType

    weight_table = WEIGHT_TABLES.get(
        classification_result.image_type,
        WEIGHT_TABLES[ImageType.UNKNOWN],
    )

    return ScoreAggregator(weights=weight_table)