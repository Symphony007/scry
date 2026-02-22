# tests/test_phase3.py

import os
import tempfile
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from core.format_handler import classify, ImageFormat, CompressionType, EmbeddingDomain
from core.dct_embedder   import _build_header, _parse_header
from core.embedder       import embed as spatial_embed
from core.decoder        import decode


# ---------------------------------------------------------------------------
# Test image helpers
# ---------------------------------------------------------------------------

def make_noise_array(height: int = 128, width: int = 128) -> np.ndarray:
    """
    Random noise image — guaranteed non-zero DCT coefficients at all
    frequencies. Used for all tests that need a realistic image.
    """
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, (height, width), dtype=np.uint8)
    return np.stack([img, img, img], axis=2)


def save_temp_image(
    arr: np.ndarray,
    suffix: str,
    quality: int = 85
) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img = Image.fromarray(arr)
    if suffix.lower() in (".jpg", ".jpeg"):
        img.save(path, format="JPEG", quality=quality)
    elif suffix.lower() == ".bmp":
        img.save(path, format="BMP")
    elif suffix.lower() == ".tiff":
        img.save(path, format="TIFF")
    else:
        img.save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------
# Group 1: Format handler (8 tests)
# ---------------------------------------------------------------------------

def test_format_handler_detects_png():
    path = save_temp_image(make_noise_array(), ".png")
    info = classify(path)
    assert info.actual_format    == ImageFormat.PNG
    assert info.compression      == CompressionType.LOSSLESS
    assert info.embedding_domain == EmbeddingDomain.SPATIAL
    os.unlink(path)


def test_format_handler_detects_jpeg():
    path = save_temp_image(make_noise_array(), ".jpg")
    info = classify(path)
    assert info.actual_format    == ImageFormat.JPEG
    assert info.compression      == CompressionType.LOSSY
    assert info.embedding_domain == EmbeddingDomain.DCT
    os.unlink(path)


def test_format_handler_detects_bmp():
    path = save_temp_image(make_noise_array(), ".bmp")
    info = classify(path)
    assert info.actual_format == ImageFormat.BMP
    assert info.compression   == CompressionType.LOSSLESS
    os.unlink(path)


def test_format_handler_detects_tiff():
    path = save_temp_image(make_noise_array(), ".tiff")
    info = classify(path)
    assert info.actual_format == ImageFormat.TIFF
    assert info.compression   == CompressionType.LOSSLESS
    os.unlink(path)


def test_format_handler_detects_webp_lossless():
    arr  = make_noise_array()
    fd, path = tempfile.mkstemp(suffix=".webp")
    os.close(fd)
    Image.fromarray(arr).save(path, format="WEBP", lossless=True)
    info = classify(path)
    assert info.actual_format == ImageFormat.WEBP
    assert info.compression   == CompressionType.LOSSLESS
    os.unlink(path)


def test_format_handler_extension_mismatch():
    arr  = make_noise_array()
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    Image.fromarray(arr).save(path, format="PNG")
    info = classify(path)
    assert info.actual_format      == ImageFormat.PNG
    assert info.extension_mismatch == True
    os.unlink(path)


def test_format_handler_image_dimensions():
    arr  = make_noise_array(80, 120)
    path = save_temp_image(arr, ".png")
    info = classify(path)
    assert info.width  == 120
    assert info.height == 80
    os.unlink(path)


def test_format_handler_file_not_found():
    with pytest.raises(Exception):
        classify("/nonexistent/path/image.png")


# ---------------------------------------------------------------------------
# Group 2: DCT header (3 tests)
# — These test pure bit manipulation, no image I/O involved
# ---------------------------------------------------------------------------

def test_dct_header_roundtrip():
    bits = _build_header(0b0010, 0b0010)
    assert len(bits) == 16
    method_id, format_code = _parse_header(bits)
    assert method_id   == 0b0010
    assert format_code == 0b0010


def test_dct_header_different_formats():
    for fmt_code in [0b0001, 0b0010, 0b0011, 0b0100]:
        bits = _build_header(0b0010, fmt_code)
        _, fc = _parse_header(bits)
        assert fc == fmt_code


def test_dct_header_too_short_raises():
    with pytest.raises(ValueError):
        _parse_header([0, 1, 0])


# ---------------------------------------------------------------------------
# Group 3: DCT embed/decode
# — Full DCT round-trips require jpegio which is not available on
#   Windows/Python 3.14 without GCC. Skipped until jpegio ships
#   a pre-built wheel for this platform.
# ---------------------------------------------------------------------------

def test_dct_rejects_png_input():
    """DCT embedder must reject PNG input with a clear error."""
    from core.dct_embedder import embed_dct
    path = save_temp_image(make_noise_array(), ".png")
    try:
        with pytest.raises(ValueError, match="DCT embedder requires"):
            embed_dct(path, "test", path.replace(".png", "_out.jpg"))
    finally:
        os.unlink(path)


def test_dct_returns_correct_structure():
    """embed_dct returns a dict with the expected keys and sane values."""
    from core.dct_embedder import embed_dct
    src = save_temp_image(make_noise_array(256, 256), ".jpg", quality=85)
    dst = src.replace(".jpg", "_stego.jpg")
    try:
        result = embed_dct(src, "test", dst)
        assert "bits_used"     in result
        assert "capacity_bits" in result
        assert "payload_pct"   in result
        assert "quality"       in result
        assert result["bits_used"]     > 0
        assert result["capacity_bits"] > result["bits_used"]
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 4: Universal decoder (8 tests)
# ---------------------------------------------------------------------------

def test_universal_decoder_png_roundtrip():
    src = save_temp_image(make_noise_array(128, 128), ".png")
    dst = src.replace(".png", "_stego.png")
    try:
        spatial_embed(src, "PNG round-trip", dst)
        result = decode(dst)
        assert result.success
        assert result.message == "PNG round-trip"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_universal_decoder_bmp_roundtrip():
    src = save_temp_image(make_noise_array(128, 128), ".bmp")
    dst = src.replace(".bmp", "_stego.bmp")
    try:
        spatial_embed(src, "BMP round-trip", dst)
        result = decode(dst)
        assert result.success
        assert result.message == "BMP round-trip"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_universal_decoder_tiff_roundtrip():
    src = save_temp_image(make_noise_array(128, 128), ".tiff")
    dst = src.replace(".tiff", "_stego.tiff")
    try:
        spatial_embed(src, "TIFF round-trip", dst)
        result = decode(dst)
        assert result.success
        assert result.message == "TIFF round-trip"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_cross_format_png_saved_as_jpeg_fails_gracefully():
    """
    A PNG embedded with LSB spatial encoding, then saved as JPEG,
    should fail gracefully — not crash.
    """
    src       = save_temp_image(make_noise_array(128, 128), ".png")
    stego_png = src.replace(".png", "_stego.png")
    stego_jpg = src.replace(".png", "_converted.jpg")
    try:
        spatial_embed(src, "will not survive", stego_png)
        Image.open(stego_png).convert("RGB").save(
            stego_jpg, format="JPEG", quality=85
        )
        result = decode(stego_jpg)
        assert result.success == False
        assert result.error   is not None
    finally:
        for p in [src, stego_png, stego_jpg]:
            if os.path.exists(p): os.unlink(p)


def test_decoder_nonexistent_file():
    result = decode("/nonexistent/file.png")
    assert result.success == False
    assert result.error   is not None


def test_decoder_clean_image_fails_gracefully():
    src    = save_temp_image(make_noise_array(64, 64), ".png")
    result = decode(src)
    assert result.success == False
    os.unlink(src)


def test_decode_with_format_hint_mismatch_warning():
    from core.decoder import decode_with_format_hint
    arr  = make_noise_array(64, 64)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    Image.fromarray(arr).save(path, format="PNG")
    result = decode_with_format_hint(path, expected_format="JPEG")
    assert any("mismatch" in w.lower() for w in result.warnings)
    os.unlink(path)