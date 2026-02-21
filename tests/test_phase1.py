# tests/test_phase1.py

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from core.utils import (
    text_to_bits,
    bits_to_text,
    calculate_capacity,
    calculate_psnr,
    load_image,
    save_image,
)
from core.embedder import embed, decode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb_image(height=64, width=64, fill=128) -> np.ndarray:
    """Create a solid-color RGB image array for testing."""
    return np.full((height, width, 3), fill, dtype=np.uint8)


def save_temp_png(array: np.ndarray) -> str:
    """Save an array to a temporary PNG file, return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    save_image(array, tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Group 1: text_to_bits / bits_to_text round-trips
# ---------------------------------------------------------------------------

def test_bits_roundtrip_ascii():
    """ASCII string survives text→bits→text round-trip."""
    original = "Hello, World!"
    assert bits_to_text(text_to_bits(original)) == original


def test_bits_roundtrip_unicode():
    """Multi-byte Unicode string survives round-trip."""
    original = "Héllo Wörld"
    assert bits_to_text(text_to_bits(original)) == original


def test_bits_roundtrip_emoji():
    """4-byte emoji survives round-trip."""
    original = "Hello 🌍"
    assert bits_to_text(text_to_bits(original)) == original


def test_bits_roundtrip_empty():
    """Empty string produces empty bit list and round-trips correctly."""
    assert text_to_bits("") == []
    assert bits_to_text([]) == ""


def test_text_to_bits_length():
    """ASCII string of N chars produces exactly N*8 bits."""
    assert len(text_to_bits("ABCD")) == 32


def test_bits_to_text_invalid_length():
    """bits_to_text raises ValueError if bit count is not a multiple of 8."""
    with pytest.raises(ValueError):
        bits_to_text([1, 0, 1])


# ---------------------------------------------------------------------------
# Group 2: calculate_capacity
# ---------------------------------------------------------------------------

def test_capacity_values():
    """Capacity calculation is correct for a known image size."""
    arr = make_rgb_image(64, 64)
    cap = calculate_capacity(arr)
    assert cap["total_bits"] == 64 * 64 * 3        # 12288
    assert cap["total_bytes"] == 64 * 64 * 3 // 8  # 1536
    assert cap["usable_bytes"] == cap["total_bytes"] - 2


def test_capacity_scales_with_size():
    """A larger image has strictly more capacity than a smaller one."""
    small = calculate_capacity(make_rgb_image(32, 32))
    large = calculate_capacity(make_rgb_image(128, 128))
    assert large["total_bits"] > small["total_bits"]


# ---------------------------------------------------------------------------
# Group 3: calculate_psnr
# ---------------------------------------------------------------------------

def test_psnr_identical_images():
    """Identical images return infinite PSNR."""
    arr = make_rgb_image()
    assert calculate_psnr(arr, arr) == float("inf")


def test_psnr_lsb_change_above_threshold():
    """LSB embedding on a solid image produces PSNR well above 48 dB."""
    original = make_rgb_image(64, 64, fill=128)
    modified = original.copy()
    # Flip every LSB — worst case LSB change
    modified[:, :, :] = (modified[:, :, :] & 0xFE) | 1
    psnr = calculate_psnr(original, modified)
    assert psnr > 48.0


def test_psnr_large_change_is_low():
    """A large pixel change produces noticeably lower PSNR than an LSB change."""
    original = make_rgb_image(64, 64, fill=100)
    badly_modified = original.copy()
    badly_modified[:, :, :] = 200  # massive change
    psnr = calculate_psnr(original, badly_modified)
    assert psnr < 20.0


# ---------------------------------------------------------------------------
# Group 4: JPEG rejection
# ---------------------------------------------------------------------------

def test_embed_rejects_jpeg(tmp_path):
    """embed() raises ValueError when given a JPEG input path."""
    fake_jpeg = tmp_path / "image.jpg"
    fake_jpeg.write_bytes(b"")  # content doesn't matter — format check is by extension
    with pytest.raises(ValueError, match="lossy"):
        embed(str(fake_jpeg), "test", str(tmp_path / "out.png"))


# ---------------------------------------------------------------------------
# Group 5: Full encode-decode round-trips
# ---------------------------------------------------------------------------

def test_roundtrip_ascii():
    """ASCII message embeds and decodes correctly."""
    arr = make_rgb_image(128, 128)
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed(src, "Hello, Scry!", dst)
        assert decode(dst) == "Hello, Scry!"
    finally:
        os.unlink(src)
        os.unlink(dst)


def test_roundtrip_unicode():
    """Unicode message embeds and decodes correctly."""
    arr = make_rgb_image(128, 128)
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed(src, "Héllo 世界 🌍", dst)
        assert decode(dst) == "Héllo 世界 🌍"
    finally:
        os.unlink(src)
        os.unlink(dst)


def test_roundtrip_empty_message():
    """Empty message embeds and decodes as empty string."""
    arr = make_rgb_image(128, 128)
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed(src, "", dst)
        assert decode(dst) == ""
    finally:
        os.unlink(src)
        os.unlink(dst)


# ---------------------------------------------------------------------------
# Group 6: Pixel change bounds
# ---------------------------------------------------------------------------

def test_pixel_delta_at_most_one():
    """No pixel channel value changes by more than 1 after embedding."""
    arr = make_rgb_image(128, 128, fill=100)
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed(src, "Delta check message", dst)
        stego, _ = load_image(dst)
        diff = np.abs(arr.astype(int) - stego.astype(int))
        assert diff.max() <= 1
    finally:
        os.unlink(src)
        os.unlink(dst)


# ---------------------------------------------------------------------------
# Group 7: Edge cases
# ---------------------------------------------------------------------------

def test_message_too_large_raises():
    """embed() raises ValueError when message exceeds image capacity."""
    arr = make_rgb_image(8, 8)  # tiny image
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        huge_message = "A" * 10000
        with pytest.raises(ValueError, match="too large"):
            embed(src, huge_message, dst)
    finally:
        os.unlink(src)
        if os.path.exists(dst):
            os.unlink(dst)


def test_embed_at_near_capacity():
    """A message near the capacity limit embeds and decodes correctly."""
    arr = make_rgb_image(128, 128)
    cap = calculate_capacity(arr)
    # usable_bytes - 2 to stay safely within bounds
    max_ascii = "Z" * (cap["usable_bytes"] - 2)
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        embed(src, max_ascii, dst)
        assert decode(dst) == max_ascii
    finally:
        os.unlink(src)
        os.unlink(dst)

def test_embed_returns_correct_structure():
    """embed() return dict contains all expected keys with sensible values."""
    arr = make_rgb_image(128, 128)
    src = save_temp_png(arr)
    dst = src.replace(".png", "_stego.png")
    try:
        result = embed(src, "Structure check", dst)
        assert "psnr" in result
        assert "bits_used" in result
        assert "capacity" in result
        assert "payload_pct" in result
        assert result["psnr"] > 48.0
        assert result["bits_used"] > 0
        assert 0 < result["payload_pct"] < 100
    finally:
        os.unlink(src)
        os.unlink(dst)