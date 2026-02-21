# core/format_handler.py

import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ImageFormat(Enum):
    PNG      = "PNG"
    JPEG     = "JPEG"
    BMP      = "BMP"
    TIFF     = "TIFF"
    WEBP     = "WEBP"
    UNKNOWN  = "UNKNOWN"


class CompressionType(Enum):
    LOSSLESS = "LOSSLESS"
    LOSSY    = "LOSSY"
    UNKNOWN  = "UNKNOWN"


class EmbeddingDomain(Enum):
    SPATIAL   = "SPATIAL"    # LSB replacement works here
    DCT       = "DCT"        # Frequency domain — JPEG, lossy WebP
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class FormatInfo:
    """
    Complete format description for an image file.
    Every field is derived from actual file bytes — never from the extension.

    Attributes:
        actual_format    : detected format from magic bytes
        compression      : LOSSLESS or LOSSY
        embedding_domain : which domain to use for embedding
        bit_depth        : bits per channel (8, 16, etc.)
        color_space      : e.g. "RGB", "RGBA", "GRAYSCALE"
        has_alpha        : True if the image has an alpha channel
        has_metadata     : True if EXIF or other metadata is present
        width            : image width in pixels
        height           : image height in pixels
        is_supported     : whether this format is supported for embedding
        extension_mismatch : True if file extension disagrees with actual format
        notes            : human-readable format summary
    """
    actual_format      : ImageFormat
    compression        : CompressionType
    embedding_domain   : EmbeddingDomain
    bit_depth          : int
    color_space        : str
    has_alpha          : bool
    has_metadata       : bool
    width              : int
    height             : int
    is_supported       : bool
    extension_mismatch : bool
    notes              : str


# ---------------------------------------------------------------------------
# Magic byte signatures for format detection
# These are read from the actual file — never trust the extension alone
# ---------------------------------------------------------------------------

MAGIC_PNG  = b'\x89PNG\r\n\x1a\n'
MAGIC_JPEG = b'\xff\xd8\xff'
MAGIC_BMP  = b'BM'
MAGIC_WEBP_RIFF = b'RIFF'
MAGIC_WEBP_WEBP = b'WEBP'  # at offset 8

# TIFF has two valid byte orders
MAGIC_TIFF_LE = b'II\x2a\x00'  # little-endian
MAGIC_TIFF_BE = b'MM\x00\x2a'  # big-endian

# Expected extensions per format
FORMAT_EXTENSIONS = {
    ImageFormat.PNG  : {".png"},
    ImageFormat.JPEG : {".jpg", ".jpeg"},
    ImageFormat.BMP  : {".bmp"},
    ImageFormat.TIFF : {".tiff", ".tif"},
    ImageFormat.WEBP : {".webp"},
}


def _read_magic(path: str, n: int = 12) -> bytes:
    """Read the first n bytes of a file for magic byte detection."""
    with open(path, "rb") as f:
        return f.read(n)


def _detect_format(magic: bytes) -> ImageFormat:
    """
    Identify image format from magic bytes.
    Order matters — check more specific signatures first.
    """
    if magic[:8] == MAGIC_PNG:
        return ImageFormat.PNG
    if magic[:3] == MAGIC_JPEG:
        return ImageFormat.JPEG
    if magic[:4] == MAGIC_WEBP_RIFF and magic[8:12] == MAGIC_WEBP_WEBP:
        return ImageFormat.WEBP
    if magic[:4] in (MAGIC_TIFF_LE, MAGIC_TIFF_BE):
        return ImageFormat.TIFF
    if magic[:2] == MAGIC_BMP:
        return ImageFormat.BMP
    return ImageFormat.UNKNOWN


def _is_webp_lossy(path: str) -> bool:
    """
    Determine if a WebP file uses lossy compression.
    WebP files contain a chunk type identifier:
        'VP8 ' (with trailing space) → lossy
        'VP8L'                       → lossless
        'VP8X'                       → extended (may be either)
    Reads bytes 12–16 to identify the chunk type.
    """
    try:
        with open(path, "rb") as f:
            f.seek(12)
            chunk_type = f.read(4)
        if chunk_type == b'VP8 ':
            return True
        if chunk_type == b'VP8L':
            return False
        # VP8X is extended format — check sub-chunks; default to lossy
        return True
    except Exception:
        return True  # safe default


def _has_exif(path: str, fmt: ImageFormat) -> bool:
    """
    Check for EXIF metadata presence.
    JPEG: look for APP1 marker (0xFFE1) in the first 64KB.
    PNG: look for 'eXIf' chunk identifier.
    Others: not checked (return False).
    """
    try:
        with open(path, "rb") as f:
            header = f.read(65536)
        if fmt == ImageFormat.JPEG:
            return b'\xff\xe1' in header
        if fmt == ImageFormat.PNG:
            return b'eXIf' in header
        return False
    except Exception:
        return False


def _read_image_dimensions(path: str, fmt: ImageFormat) -> tuple[int, int]:
    """
    Read image dimensions directly from file bytes without full decode.
    Returns (width, height).
    Falls back to (0, 0) on any error.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(32)

        if fmt == ImageFormat.PNG:
            # PNG: width at bytes 16-19, height at 20-23 (big-endian)
            w = struct.unpack(">I", data[16:20])[0]
            h = struct.unpack(">I", data[20:24])[0]
            return w, h

        if fmt == ImageFormat.JPEG:
            # JPEG dimensions require scanning for SOF marker — use Pillow
            from PIL import Image
            with Image.open(path) as img:
                return img.size  # (width, height)

        if fmt == ImageFormat.BMP:
            # BMP: width at bytes 18-21, height at 22-25 (little-endian)
            w = struct.unpack("<I", data[18:22])[0]
            h = struct.unpack("<I", data[22:26])[0]
            return w, h

        if fmt == ImageFormat.WEBP:
            # WebP: use Pillow for reliability
            from PIL import Image
            with Image.open(path) as img:
                return img.size

        if fmt == ImageFormat.TIFF:
            from PIL import Image
            with Image.open(path) as img:
                return img.size

    except Exception:
        pass

    return 0, 0


def _read_bit_depth_and_color(path: str, fmt: ImageFormat) -> tuple[int, str, bool]:
    """
    Read bit depth, color space, and alpha presence using Pillow.
    Returns (bit_depth, color_space_string, has_alpha).
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            mode = img.mode
            has_alpha = mode in ("RGBA", "LA", "PA")

            mode_to_color = {
                "RGB"  : "RGB",
                "RGBA" : "RGBA",
                "L"    : "GRAYSCALE",
                "LA"   : "GRAYSCALE+ALPHA",
                "P"    : "PALETTE",
                "PA"   : "PALETTE+ALPHA",
                "CMYK" : "CMYK",
                "YCbCr": "YCbCr",
                "I"    : "INT32",
                "F"    : "FLOAT32",
            }
            color_space = mode_to_color.get(mode, mode)

            # Bit depth per channel
            if mode in ("I",):
                bit_depth = 32
            elif mode in ("F",):
                bit_depth = 32
            else:
                bit_depth = 8  # standard for all common formats

            return bit_depth, color_space, has_alpha

    except Exception:
        return 8, "UNKNOWN", False


def classify(path: str) -> FormatInfo:
    """
    Classify an image file completely from its actual bytes.
    This is the single entry point called before any embedding or detection.

    Args:
        path: path to the image file

    Returns:
        FormatInfo with all fields populated from actual file content.

    Raises:
        FileNotFoundError: if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    magic  = _read_magic(path)
    fmt    = _detect_format(magic)
    suffix = p.suffix.lower()

    # Check for extension mismatch
    expected_exts = FORMAT_EXTENSIONS.get(fmt, set())
    extension_mismatch = bool(expected_exts) and suffix not in expected_exts

    # Determine compression type
    if fmt == ImageFormat.JPEG:
        compression = CompressionType.LOSSY
    elif fmt == ImageFormat.WEBP:
        compression = CompressionType.LOSSY if _is_webp_lossy(path) else CompressionType.LOSSLESS
    elif fmt in (ImageFormat.PNG, ImageFormat.BMP, ImageFormat.TIFF):
        compression = CompressionType.LOSSLESS
    else:
        compression = CompressionType.UNKNOWN

    # Determine embedding domain
    if fmt == ImageFormat.JPEG:
        domain = EmbeddingDomain.DCT
    elif fmt == ImageFormat.WEBP and compression == CompressionType.LOSSY:
        domain = EmbeddingDomain.DCT
    elif fmt in (ImageFormat.PNG, ImageFormat.BMP, ImageFormat.TIFF):
        domain = EmbeddingDomain.SPATIAL
    elif fmt == ImageFormat.WEBP and compression == CompressionType.LOSSLESS:
        domain = EmbeddingDomain.SPATIAL
    else:
        domain = EmbeddingDomain.UNSUPPORTED

    is_supported = fmt in (
        ImageFormat.PNG, ImageFormat.BMP, ImageFormat.TIFF,
        ImageFormat.JPEG, ImageFormat.WEBP
    )

    width, height   = _read_image_dimensions(path, fmt)
    bit_depth, color_space, has_alpha = _read_bit_depth_and_color(path, fmt)
    has_metadata    = _has_exif(path, fmt)

    notes_parts = [
        f"Format: {fmt.value}",
        f"Compression: {compression.value}",
        f"Domain: {domain.value}",
        f"Size: {width}x{height}",
        f"Color: {color_space} ({bit_depth}-bit)",
    ]
    if has_alpha:
        notes_parts.append("Has alpha channel.")
    if has_metadata:
        notes_parts.append("Metadata (EXIF) present.")
    if extension_mismatch:
        notes_parts.append(
            f"WARNING: Extension '{suffix}' does not match detected format '{fmt.value}'."
        )
    if not is_supported:
        notes_parts.append("Format not supported for embedding or detection.")

    return FormatInfo(
        actual_format      = fmt,
        compression        = compression,
        embedding_domain   = domain,
        bit_depth          = bit_depth,
        color_space        = color_space,
        has_alpha          = has_alpha,
        has_metadata       = has_metadata,
        width              = width,
        height             = height,
        is_supported       = is_supported,
        extension_mismatch = extension_mismatch,
        notes              = " | ".join(notes_parts),
    )