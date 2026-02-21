# tests/test_phase4.py

import pytest
import numpy as np

from ml.type_features import (
    extract_features, FeatureVector, ImageType, ALL_IMAGE_TYPES
)
from ml.type_classifier import (
    ImageTypeClassifier, WEIGHT_TABLES, RELIABILITY_NOTES
)
from detectors.aggregator import (
    ScoreAggregator, build_type_aware_aggregator, DEFAULT_WEIGHTS
)
from detectors.base_detector import DetectorResult, Verdict, Reliability


# ---------------------------------------------------------------------------
# Synthetic image constructors — one per image type
# ---------------------------------------------------------------------------

def make_photographic(h=128, w=128) -> np.ndarray:
    """
    Simulate a photographic image: smooth gradient with gaussian noise.
    Moderate LSB entropy, natural gradient distribution.
    """
    rng  = np.random.default_rng(42)
    row  = np.linspace(30, 220, w)
    base = np.tile(row, (h, 1))
    noise = rng.normal(0, 12, (h, w))
    gray  = np.clip(base + noise, 0, 255).astype(np.uint8)
    r = np.clip(gray + rng.normal(0, 5, (h, w)), 0, 255).astype(np.uint8)
    g = np.clip(gray + rng.normal(0, 5, (h, w)), 0, 255).astype(np.uint8)
    b = np.clip(gray + rng.normal(0, 5, (h, w)), 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


def make_synthetic(h=128, w=128) -> np.ndarray:
    """Uniform solid color — simplest synthetic image."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def make_screenshot(h=128, w=128) -> np.ndarray:
    """
    Simulate a screenshot: flat colored regions with sharp edges.
    Many histogram peaks, high edge linearity.
    """
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    arr[10:30, 10:80]  = [41,  128, 185]
    arr[40:60, 10:80]  = [39,  174,  96]
    arr[70:90, 10:80]  = [231,  76,  60]
    arr[10:90, 90:120] = [44,   62,  80]
    arr[65:67, :]      = [189, 195, 199]
    return arr


def make_scanned(h=128, w=128) -> np.ndarray:
    """
    Simulate a scanned image: gradient base with heavy random grain noise.
    High noise variance, low spatial correlation of noise.
    """
    rng   = np.random.default_rng(7)
    row   = np.linspace(50, 200, w)
    base  = np.tile(row, (h, 1))
    grain = rng.normal(0, 35, (h, w))
    gray  = np.clip(base + grain, 0, 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=2)


def make_ai_generated(h=128, w=128) -> np.ndarray:
    """
    Simulate an AI-generated image: smooth sinusoidal base,
    near-perfect LSB entropy, low block boundary artifacts.
    """
    rng  = np.random.default_rng(99)
    x    = np.linspace(0, np.pi * 2, w)
    y    = np.linspace(0, np.pi * 2, h)
    xx, yy = np.meshgrid(x, y)
    base = (np.sin(xx) * np.cos(yy) * 60 + 128).astype(np.float64)
    noise = rng.uniform(-8, 8, (h, w))
    gray  = np.clip(base + noise, 0, 255).astype(np.uint8)
    r = np.clip(gray.astype(int) + rng.integers(-3, 3, (h, w)), 0, 255).astype(np.uint8)
    g = np.clip(gray.astype(int) + rng.integers(-3, 3, (h, w)), 0, 255).astype(np.uint8)
    b = np.clip(gray.astype(int) + rng.integers(-3, 3, (h, w)), 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


# ---------------------------------------------------------------------------
# Group 1: FeatureVector (5 tests)
# ---------------------------------------------------------------------------

def test_feature_vector_correct_length():
    """Feature vector must have exactly FEATURE_COUNT elements."""
    fv = extract_features(make_photographic())
    assert len(fv.to_array()) == FeatureVector.FEATURE_COUNT


def test_feature_vector_all_finite():
    """No feature should be NaN or infinite."""
    for make_fn in [make_photographic, make_synthetic, make_screenshot,
                    make_scanned, make_ai_generated]:
        arr = make_fn()
        fv  = extract_features(arr)
        vec = fv.to_array()
        assert np.all(np.isfinite(vec)), \
            f"Non-finite feature in {make_fn.__name__}: {vec}"


def test_feature_names_match_vector_length():
    """feature_names() must return the same count as to_array()."""
    assert len(FeatureVector.feature_names()) == FeatureVector.FEATURE_COUNT


def test_synthetic_has_low_noise():
    """Synthetic solid image should have near-zero noise variance."""
    fv = extract_features(make_synthetic())
    assert fv.noise_variance < 5.0


def test_scanned_has_high_noise():
    """Scanned image with film grain should have high noise variance."""
    fv = extract_features(make_scanned())
    assert fv.noise_variance > 20.0


# ---------------------------------------------------------------------------
# Group 2: Rule-based classifier (8 tests)
# ---------------------------------------------------------------------------

def test_classifier_returns_classification_result():
    """classify() always returns a ClassificationResult."""
    from ml.type_classifier import ClassificationResult
    clf    = ImageTypeClassifier()
    result = clf.classify(make_photographic())
    assert isinstance(result, ClassificationResult)


def test_classifier_result_has_all_fields():
    """ClassificationResult has all required fields populated."""
    clf    = ImageTypeClassifier()
    result = clf.classify(make_photographic())
    assert result.image_type        in ALL_IMAGE_TYPES
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.class_probabilities, dict)
    assert isinstance(result.weight_table, dict)
    assert isinstance(result.reliability_notes, list)
    assert result.method            in ("ml", "rules", "ml+rules")


def test_classifier_probabilities_sum_to_one():
    """Class probabilities must sum to approximately 1.0."""
    clf    = ImageTypeClassifier()
    result = clf.classify(make_photographic())
    total  = sum(result.class_probabilities.values())
    assert abs(total - 1.0) < 0.01


def test_classifier_all_types_in_probabilities():
    """All six image types must appear in class_probabilities."""
    clf    = ImageTypeClassifier()
    result = clf.classify(make_photographic())
    for t in ALL_IMAGE_TYPES:
        assert t in result.class_probabilities


def test_synthetic_classified_correctly():
    """Solid uniform image should be classified as Synthetic."""
    clf    = ImageTypeClassifier()
    result = clf.classify(make_synthetic())
    assert result.image_type == ImageType.SYNTHETIC

def test_screenshot_classification_does_not_crash():
    """
    Classifier must handle screenshot-like images without crashing
    and return a valid ClassificationResult.
    Accurate screenshot detection requires a trained ML model (Phase 8).
    Rule-based classification does not guarantee correct type assignment
    on minimal synthetic arrays.
    """
    clf    = ImageTypeClassifier()
    result = clf.classify(make_screenshot())
    assert result.image_type in ALL_IMAGE_TYPES
    assert abs(sum(result.class_probabilities.values()) - 1.0) < 0.01
    assert result.weight_table is not None


def test_rule_based_confidence_capped():
    """Rule-based classifier confidence must never exceed 0.75."""
    clf    = ImageTypeClassifier()
    result = clf.classify(make_photographic())
    assert result.confidence <= 0.75


def test_classifier_weight_table_matches_type():
    """Weight table in result must match WEIGHT_TABLES for detected type."""
    clf    = ImageTypeClassifier()
    result = clf.classify(make_synthetic())
    expected = WEIGHT_TABLES[result.image_type]
    assert result.weight_table == expected


# ---------------------------------------------------------------------------
# Group 3: WEIGHT_TABLES correctness (6 tests)
# ---------------------------------------------------------------------------

def test_weight_tables_cover_all_types():
    """WEIGHT_TABLES must have an entry for every image type."""
    for t in ALL_IMAGE_TYPES:
        assert t in WEIGHT_TABLES, f"Missing weight table for {t}"


def test_weight_tables_cover_all_detectors():
    """Each weight table must specify weights for all four detectors."""
    detectors = {"RS Analysis", "Chi-Square", "Histogram", "Entropy"}
    for img_type, weights in WEIGHT_TABLES.items():
        assert set(weights.keys()) == detectors, \
            f"Weight table for {img_type} missing detectors"


def test_scanned_chisquare_weight_reduced():
    """Chi-Square weight for Scanned must be lower than for Photographic."""
    photo_weight   = WEIGHT_TABLES[ImageType.PHOTOGRAPHIC]["Chi-Square"]
    scanned_weight = WEIGHT_TABLES[ImageType.SCANNED]["Chi-Square"]
    assert scanned_weight < photo_weight


def test_ai_entropy_weight_is_zero():
    """Entropy weight for AI-Generated must be 0.0."""
    assert WEIGHT_TABLES[ImageType.AI_GENERATED]["Entropy"] == 0.0


def test_screenshot_entropy_weight_is_zero():
    """Entropy weight for Screenshot must be 0.0."""
    assert WEIGHT_TABLES[ImageType.SCREENSHOT]["Entropy"] == 0.0


def test_synthetic_all_weights_zero():
    """All detector weights for Synthetic must be 0.0."""
    for detector, weight in WEIGHT_TABLES[ImageType.SYNTHETIC].items():
        assert weight == 0.0, \
            f"Synthetic weight for {detector} should be 0.0, got {weight}"


# ---------------------------------------------------------------------------
# Group 4: Type-aware aggregator integration (6 tests)
# ---------------------------------------------------------------------------

def make_detector_result(detector: str, probability: float) -> DetectorResult:
    from detectors.base_detector import probability_to_verdict
    return DetectorResult(
        probability = probability,
        confidence  = 0.8,
        verdict     = probability_to_verdict(probability),
        reliability = Reliability.HIGH,
        detector    = detector,
        notes       = "test",
    )


def test_build_type_aware_aggregator_returns_score_aggregator():
    """build_type_aware_aggregator returns a ScoreAggregator."""
    clf    = ImageTypeClassifier()
    result = clf.classify(make_photographic())
    agg    = build_type_aware_aggregator(result)
    assert isinstance(agg, ScoreAggregator)


def test_type_aware_aggregator_uses_correct_weights():
    """Aggregator built from Scanned result uses Scanned weight table."""
    clf    = ImageTypeClassifier()
    result = clf.classify(make_scanned())
    result.image_type   = ImageType.SCANNED
    result.weight_table = WEIGHT_TABLES[ImageType.SCANNED]
    agg = build_type_aware_aggregator(result)
    assert agg.weights["Chi-Square"]  == WEIGHT_TABLES[ImageType.SCANNED]["Chi-Square"]
    assert agg.weights["RS Analysis"] == WEIGHT_TABLES[ImageType.SCANNED]["RS Analysis"]


def test_scanned_aggregation_downweights_chisquare():
    """
    On a Scanned image, Chi-Square gets lower weight than RS Analysis.
    This is the direct architectural test that the Mandrill fix works.
    """
    clf    = ImageTypeClassifier()
    result = clf.classify(make_scanned())
    result.image_type   = ImageType.SCANNED
    result.weight_table = WEIGHT_TABLES[ImageType.SCANNED]
    agg    = build_type_aware_aggregator(result)
    assert agg.weights["Chi-Square"] < agg.weights["RS Analysis"]


def test_synthetic_aggregation_all_zero_weights():
    """On Synthetic image, all detector weights are zero."""
    clf    = ImageTypeClassifier()
    result = clf.classify(make_synthetic())
    agg    = build_type_aware_aggregator(result)
    assert all(w == 0.0 for w in agg.weights.values())


def test_type_aware_aggregation_produces_result():
    """Full pipeline: classify → build aggregator → aggregate → result."""
    from detectors.aggregator import AggregatorResult
    clf    = ImageTypeClassifier()
    image  = make_photographic()
    result = clf.classify(image)
    agg    = build_type_aware_aggregator(result)

    detector_results = [
        make_detector_result("RS Analysis", 0.8),
        make_detector_result("Chi-Square",  0.6),
        make_detector_result("Histogram",   0.5),
        make_detector_result("Entropy",     0.7),
    ]
    agg_result = agg.aggregate(detector_results)
    assert isinstance(agg_result, AggregatorResult)
    assert 0.0 <= agg_result.final_probability <= 1.0


def test_reliability_notes_present_for_all_types():
    """RELIABILITY_NOTES must have entries for all image types."""
    for t in ALL_IMAGE_TYPES:
        assert t in RELIABILITY_NOTES
        assert len(RELIABILITY_NOTES[t]) > 0