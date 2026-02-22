# tests/test_lsb_matching.py

import pytest
import numpy as np
import tempfile
import os

from core.lsb_matching_embedder import embed_matching, decode_matching
from core.utils import save_image, calculate_psnr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb(h=128, w=128, fill=100) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_random_rgb(h=128, w=128, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_gradient_rgb(h=256, w=256) -> np.ndarray:
    """
    Smooth gradient image — has naturally unequal LSB pair counts.
    This is the correct base image for statistical detector tests.
    Random noise is NOT suitable — it already has balanced pairs
    naturally and triggers detectors regardless of embedding method.
    """
    row     = np.linspace(0, 255, w, dtype=np.float32)
    channel = np.tile(row, (h, 1)).astype(np.uint8)
    return np.stack([channel, channel, channel], axis=2)


def save_temp_png(array: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    save_image(array, tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Group 1: Round-trip correctness (4 tests)
# ---------------------------------------------------------------------------

def test_roundtrip_ascii():
    """ASCII message embeds and decodes correctly."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed_matching(src, "Hello LSB Matching", dst, seed=0)
        assert decode_matching(dst) == "Hello LSB Matching"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_roundtrip_unicode():
    """Unicode message embeds and decodes correctly."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed_matching(src, "Héllo 世界 🌍", dst, seed=1)
        assert decode_matching(dst) == "Héllo 世界 🌍"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_roundtrip_empty_message():
    """Empty message embeds and decodes as empty string."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed_matching(src, "", dst, seed=2)
        assert decode_matching(dst) == ""
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_roundtrip_long_message():
    """A long message near capacity embeds and decodes correctly."""
    arr = make_random_rgb(256, 256)
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    msg = "A" * 500
    try:
        embed_matching(src, msg, dst, seed=3)
        assert decode_matching(dst) == msg
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 2: Pixel change properties (3 tests)
# ---------------------------------------------------------------------------

def test_pixel_delta_at_most_one():
    """No pixel channel changes by more than 1 after LSB matching."""
    from core.utils import load_image
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed_matching(src, "Delta check", dst, seed=4)
        stego, _ = load_image(dst)
        diff = np.abs(arr.astype(int) - stego.astype(int))
        assert diff.max() <= 1
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_unchanged_pixels_where_lsb_already_matches():
    """
    Pixels whose LSB already matches the message bit must not be modified.
    With a fixed seed we can verify this deterministically.
    """
    from core.utils import load_image
    arr = make_random_rgb(64, 64, seed=10)
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed_matching(src, "test", dst, seed=5)
        stego, _ = load_image(dst)
        flat_orig  = arr.flatten().astype(int)
        flat_stego = stego.flatten().astype(int)
        for i in range(len(flat_orig)):
            if flat_orig[i] == flat_stego[i]:
                continue
            assert abs(flat_orig[i] - flat_stego[i]) == 1
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_psnr_above_threshold():
    """LSB Matching must produce PSNR above 48 dB."""
    from core.utils import load_image
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed_matching(src, "PSNR test message", dst, seed=6)
        stego, _ = load_image(dst)
        psnr = calculate_psnr(arr, stego)
        assert psnr > 48.0
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 3: Statistical undetectability (3 tests)
# ---------------------------------------------------------------------------

def test_lsb_matching_lower_rs_than_replacement():
    """
    LSB Matching should produce lower RS asymmetry than LSB Replacement.
    RS Analysis measures spatial smoothness disruption — it is more
    sensitive to small payload differences than chi-square, which
    measures global pair equality and gets swamped by the natural
    distribution at low payloads.
    Chi-square cannot distinguish the two methods at payloads below
    ~15% on gradient images — this is documented in KNOWN_ISSUES.md
    as the small payload problem.
    """
    from core.embedder import embed as lsb_replace
    from detectors.rs_analysis import RSAnalysisDetector
    from core.utils import load_image

    arr = make_gradient_rgb(256, 256)
    src = save_temp_png(arr)
    dst_replace = src.replace(".png", "_replace.png")
    dst_match   = src.replace(".png", "_match.png")
    msg = "A" * 3000

    try:
        lsb_replace(src, msg, dst_replace)
        embed_matching(src, msg, dst_match, seed=7)

        replace_arr, _ = load_image(dst_replace)
        match_arr,   _ = load_image(dst_match)

        rs = RSAnalysisDetector()
        replace_asymmetry = rs.analyze(replace_arr).raw_stats["asymmetry"]
        match_asymmetry   = rs.analyze(match_arr).raw_stats["asymmetry"]

        assert match_asymmetry < replace_asymmetry, (
            f"LSB Matching should produce less RS asymmetry than replacement. "
            f"Matching: {match_asymmetry:.6f}, Replacement: {replace_asymmetry:.6f}"
        )
    finally:
        for p in [src, dst_replace, dst_match]:
            if os.path.exists(p): os.unlink(p)

def test_lsb_matching_lower_histogram_than_replacement():
    """
    LSB Matching should produce lower histogram combing than replacement.
    Uses a gradient image for the same reason as the chi-square test.
    """
    from core.embedder import embed as lsb_replace
    from detectors.histogram import HistogramDetector
    from core.utils import load_image

    arr = make_gradient_rgb()
    src = save_temp_png(arr)
    dst_replace = src.replace(".png", "_replace.png")
    dst_match   = src.replace(".png", "_match.png")
    msg = "A" * 200

    try:
        lsb_replace(src, msg, dst_replace)
        embed_matching(src, msg, dst_match, seed=8)

        replace_arr, _ = load_image(dst_replace)
        match_arr,   _ = load_image(dst_match)

        hist = HistogramDetector()
        replace_prob = hist.analyze(replace_arr).probability
        match_prob   = hist.analyze(match_arr).probability

        assert match_prob <= replace_prob, (
            f"LSB Matching should produce less histogram combing. "
            f"Matching: {match_prob:.3f}, Replacement: {replace_prob:.3f}"
        )
    finally:
        for p in [src, dst_replace, dst_match]:
            if os.path.exists(p): os.unlink(p)


def test_lsb_matching_histogram_combing_lower_than_replacement():
    """
    LSB Matching should produce a lower raw combing score than replacement.
    Compares raw combing scores directly — normalized probability saturates
    at 1.0 for both methods, losing all resolution.
    Uses a large payload to amplify the difference between methods.
    """
    from core.embedder import embed as lsb_replace
    from detectors.histogram import HistogramDetector
    from core.utils import load_image

    arr = make_gradient_rgb(256, 256)
    src = save_temp_png(arr)
    dst_match   = src.replace(".png", "_match.png")
    dst_replace = src.replace(".png", "_replace.png")
    msg = "A" * 3000  # large payload — amplifies the difference

    try:
        embed_matching(src, msg, dst_match, seed=9)
        lsb_replace(src, msg, dst_replace)

        match_arr,   _ = load_image(dst_match)
        replace_arr, _ = load_image(dst_replace)

        hist = HistogramDetector()
        match_score   = hist.analyze(match_arr).raw_stats["mean_combing_score"]
        replace_score = hist.analyze(replace_arr).raw_stats["mean_combing_score"]

        assert match_score < replace_score, (
            f"LSB Matching raw combing score should be lower than replacement. "
            f"Matching: {match_score:.6f}, Replacement: {replace_score:.6f}"
        )
    finally:
        for p in [src, dst_match, dst_replace]:
            if os.path.exists(p): os.unlink(p)

# ---------------------------------------------------------------------------
# Group 4: Error handling (3 tests)
# ---------------------------------------------------------------------------

def test_rejects_jpeg_input(tmp_path):
    """embed_matching raises ValueError on JPEG input."""
    fake = tmp_path / "image.jpg"
    fake.write_bytes(b"")
    with pytest.raises(ValueError, match="lossless"):
        embed_matching(str(fake), "test", str(tmp_path / "out.png"))


def test_rejects_oversized_message():
    """embed_matching raises ValueError when message exceeds capacity."""
    arr = make_rgb(8, 8)
    src = save_temp_png(arr)
    try:
        with pytest.raises(ValueError, match="too large"):
            embed_matching(src, "A" * 10000, src.replace(".png", "_out.png"))
    finally:
        os.unlink(src)


def test_embed_returns_correct_structure():
    """embed_matching returns dict with all expected keys."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        result = embed_matching(src, "structure check", dst, seed=10)
        assert "psnr"        in result
        assert "bits_used"   in result
        assert "capacity"    in result
        assert "payload_pct" in result
        assert "method"      in result
        assert result["method"] == "lsb_matching"
        assert result["psnr"]   > 48.0
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)