import pytest
import numpy as np
import tempfile
import os
from PIL import Image

from core.metadata_embedder import embed_metadata, decode_metadata
from core.utils import save_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb(h=128, w=128, fill=100) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def make_random_rgb(h=128, w=128, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def save_temp(array: np.ndarray, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    img = Image.fromarray(array)
    fmt_map = {
        ".png"  : "PNG",
        ".jpg"  : "JPEG",
        ".jpeg" : "JPEG",
        ".tiff" : "TIFF",
        ".tif"  : "TIFF",
        ".bmp"  : "BMP",
    }
    img.save(tmp.name, format=fmt_map[suffix])
    return tmp.name


# ---------------------------------------------------------------------------
# Group 1: PNG round-trips (3 tests)
# ---------------------------------------------------------------------------

def test_png_roundtrip_ascii():
    """ASCII message embeds and decodes from PNG metadata."""
    arr = make_random_rgb()
    src = save_temp(arr, ".png")
    dst = src.replace(".png", "_meta.png")
    try:
        embed_metadata(src, "Hello metadata", dst)
        assert decode_metadata(dst) == "Hello metadata"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_png_roundtrip_unicode():
    """Unicode message embeds and decodes from PNG metadata."""
    arr = make_random_rgb()
    src = save_temp(arr, ".png")
    dst = src.replace(".png", "_meta.png")
    try:
        embed_metadata(src, "Héllo 世界 🌍", dst)
        assert decode_metadata(dst) == "Héllo 世界 🌍"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_png_roundtrip_long_message():
    """Long message embeds and decodes from PNG metadata."""
    arr = make_random_rgb()
    src = save_temp(arr, ".png")
    dst = src.replace(".png", "_meta.png")
    msg = "X" * 5000
    try:
        embed_metadata(src, msg, dst)
        assert decode_metadata(dst) == msg
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 2: JPEG round-trips (2 tests)
# ---------------------------------------------------------------------------

def test_jpeg_roundtrip_ascii():
    """ASCII message embeds and decodes from JPEG EXIF."""
    arr = make_random_rgb()
    src = save_temp(arr, ".jpg")
    dst = src.replace(".jpg", "_meta.jpg")
    try:
        embed_metadata(src, "JPEG metadata test", dst)
        assert decode_metadata(dst) == "JPEG metadata test"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_jpeg_roundtrip_unicode():
    """Unicode message embeds and decodes from JPEG EXIF."""
    arr = make_random_rgb()
    src = save_temp(arr, ".jpg")
    dst = src.replace(".jpg", "_meta.jpg")
    try:
        embed_metadata(src, "Héllo 世界", dst)
        assert decode_metadata(dst) == "Héllo 世界"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 3: TIFF round-trips (2 tests)
# ---------------------------------------------------------------------------

def test_tiff_roundtrip_ascii():
    """ASCII message embeds and decodes from TIFF metadata."""
    arr = make_random_rgb()
    src = save_temp(arr, ".tiff")
    dst = src.replace(".tiff", "_meta.tiff")
    try:
        embed_metadata(src, "TIFF metadata test", dst)
        assert decode_metadata(dst) == "TIFF metadata test"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_tiff_roundtrip_unicode():
    """Unicode message embeds and decodes from TIFF metadata."""
    arr = make_random_rgb()
    src = save_temp(arr, ".tiff")
    dst = src.replace(".tiff", "_meta.tiff")
    try:
        embed_metadata(src, "Héllo 世界 🌍", dst)
        assert decode_metadata(dst) == "Héllo 世界 🌍"
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 4: Zero pixel modification (3 tests)
# ---------------------------------------------------------------------------

def test_png_pixels_exactly_unchanged():
    """
    PNG metadata embedding must not change a single pixel value.
    This is the core property that distinguishes metadata from LSB methods.
    """
    arr = make_random_rgb()
    src = save_temp(arr, ".png")
    dst = src.replace(".png", "_meta.png")
    try:
        embed_metadata(src, "pixel check", dst)
        stego = np.array(Image.open(dst).convert("RGB"))
        diff  = np.abs(arr.astype(int) - stego.astype(int))
        assert diff.max() == 0, (
            f"Metadata embedding modified pixels. Max delta: {diff.max()}"
        )
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_tiff_pixels_exactly_unchanged():
    """TIFF metadata embedding must not change a single pixel value."""
    arr = make_random_rgb()
    src = save_temp(arr, ".tiff")
    dst = src.replace(".tiff", "_meta.tiff")
    try:
        embed_metadata(src, "pixel check tiff", dst)
        stego = np.array(Image.open(dst).convert("RGB"))
        diff  = np.abs(arr.astype(int) - stego.astype(int))
        assert diff.max() == 0, (
            f"Metadata embedding modified TIFF pixels. Max delta: {diff.max()}"
        )
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


def test_statistical_detectors_score_zero():
    """
    All statistical detectors must score near zero on a metadata-embedded
    image — they operate on pixel data only and must be blind to this method.
    """
    from detectors.chi_square  import ChiSquareDetector
    from detectors.entropy     import EntropyDetector
    from detectors.histogram   import HistogramDetector
    from detectors.rs_analysis import RSAnalysisDetector

    arr = make_random_rgb(256, 256)
    src = save_temp(arr, ".png")
    dst = src.replace(".png", "_meta.png")

    try:
        embed_metadata(src, "A" * 1000, dst)
        stego = np.array(Image.open(dst).convert("RGB"))

        # All detectors should produce the same result as on the clean image
        # because pixels are identical
        clean_chi  = ChiSquareDetector().analyze(arr).probability
        stego_chi  = ChiSquareDetector().analyze(stego).probability
        assert abs(clean_chi - stego_chi) < 1e-6, (
            "Chi-square result changed after metadata embedding — pixels modified."
        )

        clean_rs   = RSAnalysisDetector().analyze(arr).probability
        stego_rs   = RSAnalysisDetector().analyze(stego).probability
        assert abs(clean_rs - stego_rs) < 1e-6, (
            "RS Analysis result changed after metadata embedding — pixels modified."
        )

    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)


# ---------------------------------------------------------------------------
# Group 5: Error handling (3 tests)
# ---------------------------------------------------------------------------

def test_bmp_rejected():
    """BMP has no metadata container — must raise ValueError."""
    arr = make_random_rgb()
    src = save_temp(arr, ".bmp")
    try:
        with pytest.raises(ValueError, match=".bmp"):
            embed_metadata(src, "test", src.replace(".bmp", "_out.bmp"))
    finally:
        os.unlink(src)


def test_decode_clean_png_raises():
    """Decoding a PNG with no Scry payload must raise ValueError."""
    arr = make_random_rgb()
    src = save_temp(arr, ".png")
    try:
        with pytest.raises(ValueError, match="No Scry payload"):
            decode_metadata(src)
    finally:
        os.unlink(src)


def test_embed_returns_correct_structure():
    """embed_metadata returns dict with all expected keys."""
    arr = make_random_rgb()
    src = save_temp(arr, ".png")
    dst = src.replace(".png", "_meta.png")
    try:
        result = embed_metadata(src, "structure check", dst)
        assert "method"       in result
        assert "bits_used"    in result
        assert "pixel_delta"  in result
        assert "format"       in result
        assert result["method"]      == "metadata"
        assert result["pixel_delta"] == 0
        assert result["bits_used"]   > 0
    finally:
        for p in [src, dst]:
            if os.path.exists(p): os.unlink(p)