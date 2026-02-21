# tests/test_phase3.py

import pytest
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
from PIL import Image

from core.format_handler import (
    classify, ImageFormat, CompressionType, EmbeddingDomain
)
from core.dct_embedder import embed_dct, decode_dct, _build_header, _parse_header
from core.decoder import decode, decode_with_format_hint, DecodeResult
from core.embedder import embed
from core.utils import save_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb_array(h=128, w=128, fill=128) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_gradient_array(h=128, w=128) -> np.ndarray:
    """Smooth gradient — better for DCT than a solid fill."""
    row     = np.linspace(30, 220, w, dtype=np.uint8)
    channel = np.tile(row, (h, 1))
    return np.stack([channel, channel, channel], axis=2)


def save_temp_image(array: np.ndarray, suffix: str, quality: int = 85) -> str:
    """Save array to a temp file in the requested format. Returns path."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    img = Image.fromarray(array.astype(np.uint8), mode="RGB")
    if suffix.lower() in (".jpg", ".jpeg"):
        img.save(tmp.name, format="JPEG", quality=quality)
    elif suffix.lower() == ".png":
        img.save(tmp.name, format="PNG")
    elif suffix.lower() == ".bmp":
        img.save(tmp.name, format="BMP")
    elif suffix.lower() == ".tiff":
        img.save(tmp.name, format="TIFF")
    elif suffix.lower() == ".webp":
        img.save(tmp.name, format="WEBP", lossless=True)
    return tmp.name


def cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            os.unlink(p)


# ---------------------------------------------------------------------------
# Group 1: Format Handler — magic byte detection (8 tests)
# ---------------------------------------------------------------------------

def test_format_handler_detects_png():
    src = save_temp_image(make_gradient_array(), ".png")
    try:
        info = classify(src)
        assert info.actual_format    == ImageFormat.PNG
        assert info.compression      == CompressionType.LOSSLESS
        assert info.embedding_domain == EmbeddingDomain.SPATIAL
        assert info.is_supported     == True
    finally:
        cleanup(src)


def test_format_handler_detects_jpeg():
    src = save_temp_image(make_gradient_array(), ".jpg")
    try:
        info = classify(src)
        assert info.actual_format    == ImageFormat.JPEG
        assert info.compression      == CompressionType.LOSSY
        assert info.embedding_domain == EmbeddingDomain.DCT
        assert info.is_supported     == True
    finally:
        cleanup(src)


def test_format_handler_detects_bmp():
    src = save_temp_image(make_gradient_array(), ".bmp")
    try:
        info = classify(src)
        assert info.actual_format    == ImageFormat.BMP
        assert info.compression      == CompressionType.LOSSLESS
        assert info.embedding_domain == EmbeddingDomain.SPATIAL
    finally:
        cleanup(src)


def test_format_handler_detects_tiff():
    src = save_temp_image(make_gradient_array(), ".tiff")
    try:
        info = classify(src)
        assert info.actual_format    == ImageFormat.TIFF
        assert info.compression      == CompressionType.LOSSLESS
        assert info.embedding_domain == EmbeddingDomain.SPATIAL
    finally:
        cleanup(src)


def test_format_handler_detects_webp_lossless():
    src = save_temp_image(make_gradient_array(), ".webp")
    try:
        info = classify(src)
        assert info.actual_format    == ImageFormat.WEBP
        assert info.compression      == CompressionType.LOSSLESS
        assert info.embedding_domain == EmbeddingDomain.SPATIAL
    finally:
        cleanup(src)


def test_format_handler_extension_mismatch():
    """A JPEG file renamed to .png should trigger extension_mismatch flag."""
    jpeg_src = save_temp_image(make_gradient_array(), ".jpg")
    # Copy to a .png extension path
    fake_png = jpeg_src.replace(".jpg", "_fake.png")
    try:
        shutil.copy(jpeg_src, fake_png)
        info = classify(fake_png)
        assert info.actual_format      == ImageFormat.JPEG
        assert info.extension_mismatch == True
    finally:
        cleanup(jpeg_src, fake_png)


def test_format_handler_image_dimensions():
    """Format handler correctly reads image dimensions."""
    arr = make_gradient_array(64, 128)
    src = save_temp_image(arr, ".png")
    try:
        info = classify(src)
        assert info.width  == 128
        assert info.height == 64
    finally:
        cleanup(src)


def test_format_handler_file_not_found():
    with pytest.raises(FileNotFoundError):
        classify("nonexistent_file.png")


# ---------------------------------------------------------------------------
# Group 2: DCT Header — build and parse (3 tests)
# ---------------------------------------------------------------------------

def test_dct_header_roundtrip():
    """Header builds and parses back to original values."""
    bits = _build_header(method_id=0b0010, format_code=0b0010)
    assert len(bits) == 16
    method_id, format_code = _parse_header(bits)
    assert method_id   == 0b0010
    assert format_code == 0b0010


def test_dct_header_different_formats():
    """Header correctly encodes different format codes."""
    for fmt_code in [0b0001, 0b0010, 0b0011, 0b0100, 0b0101]:
        bits = _build_header(method_id=0b0010, format_code=fmt_code)
        _, parsed_fmt = _parse_header(bits)
        assert parsed_fmt == fmt_code


def test_dct_header_too_short_raises():
    """_parse_header raises ValueError if fewer than 16 bits provided."""
    with pytest.raises(ValueError, match="too short"):
        _parse_header([0, 1, 0])


# ---------------------------------------------------------------------------
# Group 3: DCT embed/decode round-trips (5 tests)
# ---------------------------------------------------------------------------

def test_dct_roundtrip_ascii():
    """ASCII message survives DCT embed → decode round-trip."""
    src = save_temp_image(make_gradient_array(256, 256), ".jpg", quality=85)
    dst = src.replace(".jpg", "_stego.jpg")
    try:
        embed_dct(src, "Hello DCT!", dst)
        assert decode_dct(dst) == "Hello DCT!"
    finally:
        cleanup(src, dst)


def test_dct_roundtrip_unicode():
    """Unicode message survives DCT embed → decode round-trip."""
    src = save_temp_image(make_gradient_array(256, 256), ".jpg", quality=85)
    dst = src.replace(".jpg", "_stego.jpg")
    try:
        embed_dct(src, "Héllo 世界 🌍", dst)
        assert decode_dct(dst) == "Héllo 世界 🌍"
    finally:
        cleanup(src, dst)


def test_dct_roundtrip_empty_message():
    """Empty message survives DCT round-trip."""
    src = save_temp_image(make_gradient_array(256, 256), ".jpg", quality=85)
    dst = src.replace(".jpg", "_stego.jpg")
    try:
        embed_dct(src, "", dst)
        assert decode_dct(dst) == ""
    finally:
        cleanup(src, dst)


def test_dct_rejects_png_input():
    """DCT embedder raises ValueError when given a PNG input."""
    src = save_temp_image(make_gradient_array(), ".png")
    dst = src.replace(".png", "_out.jpg")
    try:
        with pytest.raises(ValueError, match="DCT embedder requires"):
            embed_dct(src, "test", dst)
    finally:
        cleanup(src)
        cleanup(dst)


def test_dct_returns_correct_structure():
    """embed_dct returns dict with all expected keys."""
    src = save_temp_image(make_gradient_array(256, 256), ".jpg", quality=85)
    dst = src.replace(".jpg", "_stego.jpg")
    try:
        result = embed_dct(src, "Structure test", dst)
        assert "bits_used"     in result
        assert "capacity_bits" in result
        assert "payload_pct"   in result
        assert "quality"       in result
        assert result["bits_used"]   > 0
        assert result["payload_pct"] > 0
    finally:
        cleanup(src, dst)


# ---------------------------------------------------------------------------
# Group 4: JPEG survival test (1 test)
# ---------------------------------------------------------------------------

def test_dct_survives_same_quality_recompression():
    """
    A DCT-embedded message must survive one recompression cycle
    at the same quality setting used during embedding.
    """
    src      = save_temp_image(make_gradient_array(256, 256), ".jpg", quality=85)
    stego    = src.replace(".jpg", "_stego.jpg")
    recomp   = src.replace(".jpg", "_recompressed.jpg")
    try:
        embed_dct(src, "Survival test", stego)

        # Recompress at the same quality
        img = Image.open(stego)
        img.save(recomp, format="JPEG", quality=85)

        result = decode_dct(recomp)
        assert result == "Survival test"
    finally:
        cleanup(src, stego, recomp)


# ---------------------------------------------------------------------------
# Group 5: Spatial round-trips via universal decoder (4 tests)
# ---------------------------------------------------------------------------

def test_universal_decoder_png_roundtrip():
    """Universal decoder correctly routes PNG to spatial path."""
    src = save_temp_image(make_gradient_array(128, 128), ".png")
    dst = src.replace(".png", "_stego.png")
    try:
        embed(src, "PNG universal decode", dst)
        result = decode(dst)
        assert result.success        == True
        assert result.message        == "PNG universal decode"
        assert result.method_used    == "spatial"
        assert result.format_detected == "PNG"
    finally:
        cleanup(src, dst)


def test_universal_decoder_jpeg_roundtrip():
    """Universal decoder correctly routes JPEG to DCT path."""
    src = save_temp_image(make_gradient_array(256, 256), ".jpg", quality=85)
    dst = src.replace(".jpg", "_stego.jpg")
    try:
        embed_dct(src, "JPEG universal decode", dst)
        result = decode(dst)
        assert result.success         == True
        assert result.message         == "JPEG universal decode"
        assert result.method_used     == "dct"
        assert result.format_detected == "JPEG"
    finally:
        cleanup(src, dst)


def test_universal_decoder_bmp_roundtrip():
    """Universal decoder correctly routes BMP to spatial path."""
    src = save_temp_image(make_gradient_array(128, 128), ".bmp")
    dst = src.replace(".bmp", "_stego.bmp")
    try:
        embed(src, "BMP universal decode", dst)
        result = decode(dst)
        assert result.success         == True
        assert result.message         == "BMP universal decode"
        assert result.method_used     == "spatial"
        assert result.format_detected == "BMP"
    finally:
        cleanup(src, dst)


def test_universal_decoder_tiff_roundtrip():
    """Universal decoder correctly routes TIFF to spatial path."""
    src = save_temp_image(make_gradient_array(128, 128), ".tiff")
    dst = src.replace(".tiff", "_stego.tiff")
    try:
        embed(src, "TIFF universal decode", dst)
        result = decode(dst)
        assert result.success         == True
        assert result.message         == "TIFF universal decode"
        assert result.method_used     == "spatial"
        assert result.format_detected == "TIFF"
    finally:
        cleanup(src, dst)


# ---------------------------------------------------------------------------
# Group 6: Cross-format failure handling (4 tests)
# ---------------------------------------------------------------------------

def test_cross_format_png_saved_as_jpeg_fails_gracefully():
    """
    Embedding in PNG then saving as JPEG destroys the message.
    The decoder must return success=False with a clear explanation —
    not garbage or a raw exception.
    """
    src      = save_temp_image(make_gradient_array(128, 128), ".png")
    stego    = src.replace(".png", "_stego.png")
    jpeg_out = src.replace(".png", "_converted.jpg")
    try:
        embed(src, "Cross format test", stego)

        # Convert stego PNG to JPEG — destroys LSB embedding
        img = Image.open(stego).convert("RGB")
        img.save(jpeg_out, format="JPEG", quality=85)

        # DCT decoder should fail gracefully — not crash
        result = decode(jpeg_out)
        assert result.success == False
        assert len(result.error) > 0
    finally:
        cleanup(src, stego, jpeg_out)


def test_decoder_nonexistent_file():
    """Decoder returns structured failure for missing files."""
    result = decode("this_file_does_not_exist.png")
    assert result.success == False
    assert "not found" in result.error.lower()


def test_decoder_clean_image_fails_gracefully():
    """
    Decoding a clean image with no hidden message must return
    success=False with a clear explanation — not an exception.
    """
    src = save_temp_image(make_gradient_array(128, 128), ".png")
    try:
        result = decode(src)
        assert result.success == False
        assert len(result.error) > 0
    finally:
        cleanup(src)


def test_decode_with_format_hint_mismatch_warning():
    """
    decode_with_format_hint adds a warning when detected format
    differs from the expected format hint.
    """
    src   = save_temp_image(make_gradient_array(128, 128), ".png")
    stego = src.replace(".png", "_stego.png")
    try:
        embed(src, "Hint test", stego)
        # Tell decoder to expect JPEG — but file is PNG
        result = decode_with_format_hint(stego, expected_format="JPEG")
        mismatch_warnings = [w for w in result.warnings if "mismatch" in w.lower()]
        assert len(mismatch_warnings) > 0
    finally:
        cleanup(src, stego)