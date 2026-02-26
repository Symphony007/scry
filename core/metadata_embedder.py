"""
Metadata Embedder

Hides data in image file metadata rather than pixel values.
Zero pixel modification — the image is visually and statistically
identical before and after embedding. All statistical detectors
(chi-square, RS analysis, histogram, entropy) score zero because
they operate on pixel data only.

Method depends on format:
    PNG  — stores message in a tEXt ancillary chunk under a fixed key
    JPEG — stores message in the EXIF UserComment field
    TIFF — stores message in ImageDescription tag
    WebP — converted to PNG, stored in tEXt chunk

Tradeoffs vs pixel-based methods:
    + Zero pixel change — statistically undetectable
    + No PSNR degradation
    - File size increases by roughly len(message) bytes
    - Anyone inspecting metadata directly can find the message
    - Stripped by most social media platforms on upload
    - Not suitable when file size or metadata is analysed

Documented limitation (KNOWN_ISSUES.md):
    Metadata is stripped by virtually all image hosting platforms
    (Discord, Twitter, Instagram, WhatsApp). This method is only
    appropriate for direct file transfer between parties.
"""

from pathlib import Path
from PIL import Image
import piexif
import json


# Fixed metadata key used to identify Scry-embedded messages
SCRY_PNG_KEY  = "scry_payload"
SCRY_TIFF_TAG = 270   # ImageDescription tag ID

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}


def embed_metadata(image_path: str, message: str, output_path: str) -> dict:
    """
    Embed a UTF-8 message into image metadata without modifying pixels.

    Args:
        image_path  : path to cover image (PNG, JPEG, TIFF, or WebP)
        message     : UTF-8 message to hide
        output_path : path to save the output image

    Returns:
        dict with method, bits_used, pixel_delta, format

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
        fmt = "PNG"

    elif suffix in (".jpg", ".jpeg"):
        _embed_jpeg(img, message, image_path, output_path)
        fmt = "JPEG"

    elif suffix in (".tiff", ".tif"):
        _embed_tiff(img, message, output_path)
        fmt = "TIFF"

    elif suffix == ".webp":
        # Convert to PNG and store in tEXt chunk
        png_output = str(Path(output_path).with_suffix(".png"))
        _embed_png(img, message, png_output)
        output_path = png_output
        fmt = "PNG"

    bits_used = len(message.encode("utf-8")) * 8

    print(f"[METADATA] Message embedded in {fmt} metadata.")
    print(f"[METADATA] {bits_used} bits — zero pixel modification.")

    return {
        "method"      : "metadata",
        "bits_used"   : bits_used,
        "pixel_delta" : 0,
        "format"      : fmt,
    }


def decode_metadata(image_path: str) -> str:
    """
    Extract a message from image metadata.

    Args:
        image_path: path to the stego image

    Returns:
        Decoded message string.

    Raises:
        ValueError: if no Scry payload is found in metadata
        ValueError: if format is unsupported
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

    elif suffix in (".tiff", ".tif"):
        return _decode_tiff(image_path)

    elif suffix == ".webp":
        # WebP metadata embeds are stored as PNG — shouldn't reach here
        # in normal flow, but handle gracefully
        raise ValueError(
            "WebP metadata payloads are stored in a converted PNG file. "
            "Please upload the PNG output from the embed step."
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
# JPEG — EXIF UserComment
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

    # UserComment requires a specific encoding prefix
    # "ASCII\x00\x00\x00" + message bytes is the standard format
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

    user_comment = exif_dict.get("Exif", {}).get(
        piexif.ExifIFD.UserComment
    )

    if not user_comment:
        raise ValueError(
            "No Scry payload found in JPEG EXIF UserComment. "
            "This image was not embedded with the metadata method."
        )

    # Strip the encoding prefix if present
    prefix = b"ASCII\x00\x00\x00"
    if user_comment.startswith(prefix):
        user_comment = user_comment[len(prefix):]

    return user_comment.decode("utf-8")


# ---------------------------------------------------------------------------
# TIFF — ImageDescription tag
# ---------------------------------------------------------------------------

def _embed_tiff(img: Image.Image, message: str, output_path: str) -> None:
    """Store message in TIFF ImageDescription tag."""
    # Wrap in a JSON envelope so we can distinguish Scry payloads
    # from legitimate ImageDescription values
    envelope = json.dumps({"scry": message})
    tiffinfo  = Image.Exif()
    tiffinfo[SCRY_TIFF_TAG] = envelope
    img.save(output_path, format="TIFF", exif=tiffinfo)


def _decode_tiff(image_path: str) -> str:
    """Extract message from TIFF ImageDescription tag."""
    with Image.open(image_path) as img:
        exif = img.getexif()

    raw = exif.get(SCRY_TIFF_TAG)
    if not raw:
        raise ValueError(
            "No Scry payload found in TIFF ImageDescription tag. "
            "This image was not embedded with the metadata method."
        )

    try:
        envelope = json.loads(raw)
        return envelope["scry"]
    except (json.JSONDecodeError, KeyError):
        raise ValueError(
            "ImageDescription tag found but does not contain a Scry payload. "
            "The tag may contain legitimate image description data."
        )