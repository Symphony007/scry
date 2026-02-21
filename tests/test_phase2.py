# tests/test_phase2.py

import pytest
import numpy as np
import tempfile
import os

from detectors.base_detector import (
    DetectorResult, Verdict, Reliability, probability_to_verdict
)
from detectors.chi_square  import ChiSquareDetector
from detectors.entropy     import EntropyDetector
from detectors.rs_analysis import RSAnalysisDetector
from detectors.histogram   import HistogramDetector
from detectors.aggregator  import ScoreAggregator, DEFAULT_WEIGHTS
from visualizers.lsb_plane import LSBVisualizer
from core.utils import save_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb(h=128, w=128, fill=128) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_gradient_rgb(h=128, w=128) -> np.ndarray:
    """Smooth horizontal gradient — gives RS analysis something to measure."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    channel = np.tile(row, (h, 1))
    return np.stack([channel, channel, channel], axis=2)


def make_random_rgb(h=128, w=128, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_combed_rgb(h=128, w=128) -> np.ndarray:
    """
    Image where every adjacent pixel pair (0,1),(2,3)... has equal counts.
    Simulates perfect LSB replacement histogram combing.
    """
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    flat = arr.reshape(-1)
    for i in range(len(flat)):
        flat[i] = (i % 256) & 0xFE  # all even values → will be combed
    # Force even/odd equality by alternating
    for i in range(0, len(flat) - 1, 2):
        flat[i]     = (i % 256) & 0xFE        # even value
        flat[i + 1] = flat[i] | 1              # its odd neighbor
    return flat.reshape(h, w, 3)


def save_temp_png(array: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    save_image(array, tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Group 1: DetectorResult validation (4 tests)
# ---------------------------------------------------------------------------

def test_detector_result_valid():
    r = DetectorResult(
        probability=0.8, confidence=0.9,
        verdict=Verdict.STEGO, reliability=Reliability.HIGH,
        detector="Test", notes="ok"
    )
    assert r.probability == 0.8
    assert r.verdict == Verdict.STEGO


def test_detector_result_invalid_probability():
    with pytest.raises(ValueError):
        DetectorResult(
            probability=1.5, confidence=0.5,
            verdict=Verdict.CLEAN, reliability=Reliability.HIGH,
            detector="Test", notes="bad"
        )


def test_detector_result_invalid_confidence():
    with pytest.raises(ValueError):
        DetectorResult(
            probability=0.5, confidence=-0.1,
            verdict=Verdict.CLEAN, reliability=Reliability.HIGH,
            detector="Test", notes="bad"
        )


def test_probability_to_verdict_thresholds():
    assert probability_to_verdict(0.0)  == Verdict.CLEAN
    assert probability_to_verdict(0.39) == Verdict.CLEAN
    assert probability_to_verdict(0.40) == Verdict.SUSPICIOUS
    assert probability_to_verdict(0.69) == Verdict.SUSPICIOUS
    assert probability_to_verdict(0.70) == Verdict.STEGO
    assert probability_to_verdict(1.0)  == Verdict.STEGO


# ---------------------------------------------------------------------------
# Group 2: Chi-Square detector (7 tests)
# ---------------------------------------------------------------------------

def test_chisquare_clean_image_low_probability():
    """
    All-even-pixel image has maximally unequal pairs (all counts in even slot).
    Chi-square should detect this as NOT stego — low probability.
    """
    arr = make_rgb(128, 128, fill=128)  # all pixels = 128 (even)
    result = ChiSquareDetector().analyze(arr)
    assert isinstance(result, DetectorResult)
    assert result.probability < 0.5


def test_chisquare_combed_image_high_probability():
    """
    Perfectly combed image (equal even/odd pairs) should score high probability.
    """
    arr = make_combed_rgb(128, 128)
    result = ChiSquareDetector().analyze(arr)
    assert result.probability > 0.4


def test_chisquare_returns_detector_result():
    result = ChiSquareDetector().analyze(make_gradient_rgb())
    assert isinstance(result, DetectorResult)
    assert result.detector == "Chi-Square"


def test_chisquare_probability_in_range():
    result = ChiSquareDetector().analyze(make_random_rgb())
    assert 0.0 <= result.probability <= 1.0


def test_chisquare_has_raw_stats():
    result = ChiSquareDetector().analyze(make_gradient_rgb())
    assert "p_value" in result.raw_stats


def test_chisquare_probability_equals_pvalue():
    """probability must equal p_value directly — never 1 - p_value."""
    result = ChiSquareDetector().analyze(make_gradient_rgb())
    assert abs(result.probability - result.raw_stats["p_value"]) < 1e-9


def test_chisquare_tiny_image_does_not_crash():
    arr = make_rgb(4, 4, fill=100)
    result = ChiSquareDetector().analyze(arr)
    assert isinstance(result, DetectorResult)


# ---------------------------------------------------------------------------
# Group 3: Entropy detector (7 tests)
# ---------------------------------------------------------------------------

def test_entropy_low_on_uniform_image():
    """
    A uniform solid image has zero LSB entropy — all LSBs are the same.
    Probability should be 0.
    """
    arr = make_rgb(128, 128, fill=128)  # all even → all LSBs = 0
    result = EntropyDetector().analyze(arr)
    assert result.probability == 0.0


def test_entropy_high_on_random_image():
    """Random noise image has near-maximum LSB entropy."""
    arr = make_random_rgb(128, 128)
    result = EntropyDetector().analyze(arr)
    assert result.raw_stats["mean_entropy"] > 0.7


def test_entropy_returns_detector_result():
    result = EntropyDetector().analyze(make_gradient_rgb())
    assert isinstance(result, DetectorResult)
    assert result.detector == "Entropy"


def test_entropy_probability_in_range():
    result = EntropyDetector().analyze(make_random_rgb())
    assert 0.0 <= result.probability <= 1.0


def test_entropy_has_raw_stats():
    result = EntropyDetector().analyze(make_gradient_rgb())
    assert "mean_entropy" in result.raw_stats
    assert "block_count" in result.raw_stats


def test_entropy_confidence_is_low():
    """Entropy detector is a supporting signal — confidence must be <= 0.5."""
    result = EntropyDetector().analyze(make_gradient_rgb())
    assert result.confidence <= 0.5


def test_entropy_tiny_image_does_not_crash():
    arr = make_rgb(4, 4, fill=200)
    result = EntropyDetector().analyze(arr)
    assert isinstance(result, DetectorResult)


# ---------------------------------------------------------------------------
# Group 4: RS Analysis detector (7 tests)
# ---------------------------------------------------------------------------

def test_rs_clean_gradient_low_probability():
    """
    A smooth gradient image should show low RS asymmetry — clean signal.
    """
    arr = make_gradient_rgb(128, 128)
    result = RSAnalysisDetector().analyze(arr)
    assert isinstance(result, DetectorResult)
    assert result.raw_stats["asymmetry"] < 0.20


def test_rs_returns_detector_result():
    result = RSAnalysisDetector().analyze(make_gradient_rgb())
    assert isinstance(result, DetectorResult)
    assert result.detector == "RS Analysis"


def test_rs_probability_in_range():
    result = RSAnalysisDetector().analyze(make_gradient_rgb())
    assert 0.0 <= result.probability <= 1.0


def test_rs_has_raw_stats():
    result = RSAnalysisDetector().analyze(make_gradient_rgb())
    for key in ["asymmetry", "rm", "sm", "r_m", "s_m", "payload_estimate_pct"]:
        assert key in result.raw_stats


def test_rs_payload_estimate_is_non_negative():
    result = RSAnalysisDetector().analyze(make_gradient_rgb())
    assert result.raw_stats["payload_estimate_pct"] >= 0.0


def test_rs_confidence_is_high():
    """RS Analysis is the most reliable detector — confidence must be >= 0.8."""
    result = RSAnalysisDetector().analyze(make_gradient_rgb())
    assert result.confidence >= 0.8


def test_rs_tiny_image_does_not_crash():
    arr = make_gradient_rgb(16, 16)
    result = RSAnalysisDetector().analyze(arr)
    assert isinstance(result, DetectorResult)


# ---------------------------------------------------------------------------
# Group 5: Histogram detector (5 tests)
# ---------------------------------------------------------------------------

def test_histogram_combed_image_high_score():
    """Perfectly combed image should produce high combing score."""
    arr = make_combed_rgb(128, 128)
    result = HistogramDetector().analyze(arr)
    assert result.raw_stats["mean_combing_score"] > 0.7


def test_histogram_returns_detector_result():
    result = HistogramDetector().analyze(make_gradient_rgb())
    assert isinstance(result, DetectorResult)
    assert result.detector == "Histogram"


def test_histogram_probability_in_range():
    result = HistogramDetector().analyze(make_random_rgb())
    assert 0.0 <= result.probability <= 1.0


def test_histogram_has_channel_scores():
    result = HistogramDetector().analyze(make_gradient_rgb())
    scores = result.raw_stats["channel_scores"]
    assert "R" in scores and "G" in scores and "B" in scores


def test_histogram_tiny_image_does_not_crash():
    arr = make_rgb(4, 4, fill=50)
    result = HistogramDetector().analyze(arr)
    assert isinstance(result, DetectorResult)


# ---------------------------------------------------------------------------
# Group 6: Aggregator (5 tests)
# ---------------------------------------------------------------------------

def test_aggregator_default_weights():
    """Aggregator uses DEFAULT_WEIGHTS when none are provided."""
    agg = ScoreAggregator()
    assert agg.weights == DEFAULT_WEIGHTS


def test_aggregator_custom_weights():
    """Aggregator accepts and uses custom weight dictionary."""
    custom = {"RS Analysis": 1.0, "Chi-Square": 0.0, "Histogram": 0.0, "Entropy": 0.0}
    agg = ScoreAggregator(weights=custom)
    assert agg.weights["Chi-Square"] == 0.0


def test_aggregator_empty_results():
    agg = ScoreAggregator()
    result = agg.aggregate([])
    assert result.final_probability == 0.0
    assert result.final_verdict == Verdict.CLEAN


def test_aggregator_zero_weight_excluded():
    """A detector with weight 0.0 must not influence the final probability."""
    high_result = DetectorResult(
        probability=1.0, confidence=1.0,
        verdict=Verdict.STEGO, reliability=Reliability.HIGH,
        detector="Chi-Square", notes="high"
    )
    low_result = DetectorResult(
        probability=0.0, confidence=1.0,
        verdict=Verdict.CLEAN, reliability=Reliability.HIGH,
        detector="RS Analysis", notes="low"
    )
    # Give Chi-Square weight 0 — should not affect result
    weights = {"RS Analysis": 1.0, "Chi-Square": 0.0, "Histogram": 0.0, "Entropy": 0.0}
    agg = ScoreAggregator(weights=weights)
    result = agg.aggregate([high_result, low_result])
    assert result.final_probability == 0.0


def test_aggregator_weighted_combination():
    """Final probability reflects weighted average of detector probabilities."""
    r1 = DetectorResult(
        probability=1.0, confidence=1.0,
        verdict=Verdict.STEGO, reliability=Reliability.HIGH,
        detector="RS Analysis", notes=""
    )
    r2 = DetectorResult(
        probability=0.0, confidence=1.0,
        verdict=Verdict.CLEAN, reliability=Reliability.HIGH,
        detector="Chi-Square", notes=""
    )
    # RS=2.0, Chi=1.0 → expected = (1.0*2 + 0.0*1) / 3 = 0.666...
    weights = {"RS Analysis": 2.0, "Chi-Square": 1.0, "Histogram": 0.0, "Entropy": 0.0}
    agg = ScoreAggregator(weights=weights)
    result = agg.aggregate([r1, r2])
    assert abs(result.final_probability - (2.0 / 3.0)) < 1e-6


# ---------------------------------------------------------------------------
# Group 7: LSB Visualizer (2 tests)
# ---------------------------------------------------------------------------

def test_visualizer_lsb_plane_saves_file():
    arr = make_gradient_rgb(128, 128)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        LSBVisualizer().save_lsb_plane(arr, tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.unlink(tmp.name)


def test_visualizer_comparison_saves_file():
    original = make_gradient_rgb(128, 128)
    stego    = original.copy()
    stego[:, :, 0] = (stego[:, :, 0] & 0xFE) | 1  # flip all R LSBs
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        LSBVisualizer().save_comparison(original, stego, tmp.name)
        assert os.path.exists(tmp.name)
        assert os.path.getsize(tmp.name) > 0
    finally:
        os.unlink(tmp.name)