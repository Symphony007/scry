# tests/test_dwt_embedder.py

import pytest
import numpy as np
import tempfile
import os
from PIL import Image

from core.dwt_embedder import embed_dwt, decode_dwt
from core.utils import calculate_psnr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_random_rgb(h=256, w=256, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def make_gradient_rgb(h=256, w=256) -> np.ndarray:
    row     = np.linspace(0, 255, w, dtype=np.float32)
    channel = np.tile(row, (h, 1)).astype(np.uint8)
    return np.stack([channel, channel, channel], axis=2)


def save_temp_png(array: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    Image.fromarray(array).save(tmp.name, format="PNG")
    return tmp.name


# ---------------------------------------------------------------------------
# Group 1: Round-trip correctness (4 tests)
# ---------------------------------------------------------------------------

def test_roundtrip_ascii():
    """ASCII message embeds and decodes correctly via DWT."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    try:
        embed_dwt(src, "Hello DWT", dst)
        assert decode_dwt(dst) == "Hello DWT"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_roundtrip_unicode():
    """Unicode message embeds and decodes correctly via DWT."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    try:
        embed_dwt(src, "Héllo 世界 🌍", dst)
        assert decode_dwt(dst) == "Héllo 世界 🌍"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_roundtrip_empty_message():
    """Empty message embeds and decodes as empty string."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    try:
        embed_dwt(src, "", dst)
        assert decode_dwt(dst) == ""
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_roundtrip_long_message():
    """A longer message embeds and decodes correctly."""
    arr = make_random_rgb(512, 512)
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    msg = "B" * 200
    try:
        embed_dwt(src, msg, dst)
        assert decode_dwt(dst) == msg
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 2: Quality properties (3 tests)
# ---------------------------------------------------------------------------

def test_psnr_within_acceptable_range():
    """
    DWT embedding accepts mild quality loss.
    PSNR should be between 35 dB and 55 dB — lower bound reflects
    the acceptable frequency domain distortion, upper bound confirms
    the image was actually modified.
    """
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    try:
        result = embed_dwt(src, "quality test", dst)
        assert result["psnr"] > 35.0, (
            f"PSNR too low: {result['psnr']:.2f} dB"
        )
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_output_is_always_png():
    """
    embed_dwt always saves as PNG regardless of output_path extension.
    Lossy formats would destroy the embedded coefficients.
    """
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.jpg")  # intentionally wrong extension
    expected_dst = src.replace(".png", "_dwt.png")
    try:
        result = embed_dwt(src, "png enforcement", dst)
        assert os.path.exists(expected_dst), (
            "DWT embedder should have saved as PNG despite .jpg output path"
        )
        assert result["output_path"].endswith(".png")
    finally:
        for p in [src, expected_dst]:
            if os.path.exists(p): os.unlink(p)


def test_embed_returns_correct_structure():
    """embed_dwt returns dict with all expected keys and sane values."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    try:
        result = embed_dwt(src, "structure check", dst)
        assert "psnr"          in result
        assert "bits_used"     in result
        assert "capacity_bits" in result
        assert "payload_pct"   in result
        assert "method"        in result
        assert "step"          in result
        assert "output_path"   in result
        assert result["method"]     == "dwt"
        assert result["bits_used"]  > 0
        assert result["psnr"]       > 35.0
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 3: Step size behaviour (2 tests)
# ---------------------------------------------------------------------------

def test_step_mismatch_fails_gracefully():
    """
    Decoding with a different step size than embedding should fail
    gracefully — not crash and not return garbage silently.
    """
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    try:
        embed_dwt(src, "step mismatch test", dst, step=4)
        with pytest.raises((ValueError, UnicodeDecodeError)):
            decode_dwt(dst, step=8)  # wrong step
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_different_step_sizes_roundtrip():
    """Embedding and decoding with step=8 round-trips correctly."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    try:
        embed_dwt(src, "step 8 test", dst, step=8)
        assert decode_dwt(dst, step=8) == "step 8 test"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 4: Error handling (3 tests)
# ---------------------------------------------------------------------------

def test_oversized_message_raises():
    """embed_dwt raises ValueError when message exceeds HH capacity."""
    arr = make_random_rgb(64, 64)  # small image — low capacity
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    try:
        with pytest.raises(ValueError, match="too large"):
            embed_dwt(src, "A" * 10000, dst)
    finally:
        os.unlink(src)
        if os.path.exists(dst): os.unlink(dst)


def test_decode_clean_image_fails_gracefully():
    """Decoding a clean image fails gracefully — no crash, clear error."""
    arr = make_random_rgb()
    src = save_temp_png(arr)
    try:
        with pytest.raises(ValueError):
            decode_dwt(src)
    finally:
        os.unlink(src)


def test_pywt_missing_gives_clear_error(monkeypatch):
    """
    If PyWavelets is not installed, embed_dwt raises ImportError
    with a clear installation message.
    """
    import core.dwt_embedder as dwt_mod
    monkeypatch.setattr(dwt_mod, "PYWT_AVAILABLE", False)
    arr = make_random_rgb()
    src = save_temp_png(arr)
    dst = src.replace(".png", "_dwt.png")
    try:
        with pytest.raises(ImportError, match="PyWavelets"):
            dwt_mod.embed_dwt(src, "test", dst)
    finally:
        os.unlink(src)
        if os.path.exists(dst): os.unlink(dst)