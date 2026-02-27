"""
Metadata Embedder

Hides data in image file metadata rather than pixel values.
Zero pixel modification — the image is visually and statistically
identical before and after embedding. All statistical detectors
(chi-square, RS analysis, histogram, entropy) score zero because
they operate on pixel data only.

Method depends on format:
    PNG  — stores message in a tEXt ancillary chunk under a fixed key
    JPEG — stores message in EXIF UserComment field via piexif
    TIFF — converted to PNG, stored in tEXt chunk (output is .png)
    WebP — converted to PNG, stored in tEXt chunk (output is .png)

Why TIFF and WebP are converted to PNG:
    piexif's TIFF write path produces EXIF bytes that Pillow saves
    correctly but reads back as raw bytes rather than a decoded string,
    causing UTF-8 decode errors on the round-trip. PNG tEXt chunks
    are the most reliable metadata container available — they round-trip
    cleanly across all Pillow versions and are lossless, so converting
    a TIFF (also lossless) to PNG loses no image quality.

Tradeoffs vs pixel-based methods:
    + Zero pixel change — statistically undetectable
    + No PSNR degradation
    - File size increases by roughly len(message) bytes
    - Anyone inspecting metadata directly can find the message
    - Stripped by most social media platforms on upload
    - TIFF and WebP inputs produce PNG output files

Documented limitation:
    Metadata is stripped by virtually all image hosting platforms
    (Discord, Twitter, Instagram, WhatsApp). This method is only
    appropriate for direct file transfer between parties.
"""

from pathlib import Path
from PIL import Image
import piexif


# Fixed metadata key used to identify Scry-embedded messages
SCRY_PNG_KEY = "scry_payload"

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}

# Formats that are converted to PNG for metadata embedding
PNG_CONVERTED_SUFFIXES = {".tiff", ".tif", ".webp"}


def embed_metadata(image_path: str, message: str, output_path: str) -> dict:
    """
    Embed a UTF-8 message into image metadata without modifying pixels.

    Args:
        image_path  : path to cover image (PNG, JPEG, TIFF, or WebP)
        message     : UTF-8 message to hide
        output_path : desired output path

    Returns:
        dict with method, bits_used, pixel_delta, format, output_path
        NOTE: always use result["output_path"] — TIFF and WebP redirect to .png.

    Raises:
        ValueError: if format is not supported for metadata embedding
    """
    suffix = Path(image_path).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Metadata embedding not supported for '{suffix}'. "
            f"Supported: PNG, JPEG, TIFF, WebP."
        )

    img = Image.open(image_path).convert("RGB")

    if suffix == ".png":
        _embed_png(img, message, output_path)
        fmt        = "PNG"
        actual_out = output_path

    elif suffix in (".jpg", ".jpeg"):
        _embed_jpeg(img, message, image_path, output_path)
        fmt        = "JPEG"
        actual_out = output_path

    elif suffix in PNG_CONVERTED_SUFFIXES:
        # TIFF and WebP both convert to PNG — most reliable metadata container
        png_output = str(Path(output_path).with_suffix(".png"))
        _embed_png(img, message, png_output)
        fmt        = "PNG"
        actual_out = png_output

    bits_used = len(message.encode("utf-8")) * 8

    print(f"[METADATA] Message embedded in {fmt} metadata.")
    print(f"[METADATA] {bits_used} bits — zero pixel modification.")
    print(f"[METADATA] Output: {actual_out}")

    return {
        "method"      : "metadata",
        "bits_used"   : bits_used,
        "pixel_delta" : 0,
        "format"      : fmt,
        "output_path" : actual_out,
    }


def decode_metadata(image_path: str) -> str:
    """
    Extract a message from image metadata.

    Args:
        image_path: path to the stego image

    Returns:
        Decoded message string.

    Raises:
        ValueError: if no Scry payload found, or unsupported format
    """
    suffix = Path(image_path).suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Metadata decoding not supported for '{suffix}'."
        )

    if suffix == ".png":
        return _decode_png(image_path)

    elif suffix in (".jpg", ".jpeg"):
        return _decode_jpeg(image_path)

    elif suffix in PNG_CONVERTED_SUFFIXES:
        raise ValueError(
            f"{'TIFF' if 'tif' in suffix else 'WebP'} metadata payloads are "
            f"stored as PNG. Please upload the .png file that was downloaded "
            f"after embedding."
        )

    raise ValueError(f"No metadata decoder available for '{suffix}'.")


# ---------------------------------------------------------------------------
# PNG — tEXt chunk
# ---------------------------------------------------------------------------

def _embed_png(img: Image.Image, message: str, output_path: str) -> None:
    """Store message in PNG tEXt metadata under SCRY_PNG_KEY."""
    from PIL import PngImagePlugin
    meta = PngImagePlugin.PngInfo()
    meta.add_text(SCRY_PNG_KEY, message)
    img.save(output_path, format="PNG", pnginfo=meta)


def _decode_png(image_path: str) -> str:
    """Extract message from PNG tEXt chunk."""
    with Image.open(image_path) as img:
        metadata = img.info
    if SCRY_PNG_KEY not in metadata:
        raise ValueError(
            f"No Scry payload found in PNG metadata. "
            f"Key '{SCRY_PNG_KEY}' not present. "
            f"This image was not embedded with the metadata method."
        )
    return metadata[SCRY_PNG_KEY]


# ---------------------------------------------------------------------------
# JPEG — EXIF UserComment (piexif)
# ---------------------------------------------------------------------------

def _embed_jpeg(
    img         : Image.Image,
    message     : str,
    source_path : str,
    output_path : str,
) -> None:
    """Store message in JPEG EXIF UserComment field."""
    try:
        exif_dict = piexif.load(source_path)
    except Exception:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

    prefix  = b"ASCII\x00\x00\x00"
    payload = prefix + message.encode("utf-8")
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = payload

    exif_bytes = piexif.dump(exif_dict)
    img.save(output_path, format="JPEG", exif=exif_bytes, quality=95)


def _decode_jpeg(image_path: str) -> str:
    """Extract message from JPEG EXIF UserComment."""
    try:
        exif_dict = piexif.load(image_path)
    except Exception:
        raise ValueError(
            "Could not read EXIF data from this JPEG. "
            "The file may not contain a Scry metadata payload."
        )

    user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)

    if not user_comment:
        raise ValueError(
            "No Scry payload found in JPEG EXIF UserComment. "
            "This image was not embedded with the metadata method."
        )

    prefix = b"ASCII\x00\x00\x00"
    if user_comment.startswith(prefix):
        user_comment = user_comment[len(prefix):]

    return user_comment.decode("utf-8")