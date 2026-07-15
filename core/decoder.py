"""
core/decoder.py — Universal multi-method decoder

Purpose:
    Single entry point for all decoding operations.
    The caller never needs to know which embedding method was used —
    the decoder tries all known methods in priority order until one succeeds.

Inputs:
    A file path to any supported image (PNG, JPEG, TIFF, WebP).

Outputs:
    A DecodeResult object containing: success flag, the decoded message,
    which method succeeded, format info, and any warnings.

Responsibilities:
    - Classify the image format first
    - Try each decode method in order: metadata -> DWT -> spatial LSB -> DCT
    - Return structured results for every failure mode (never raw exceptions)

How it fits in:
    Called by web/app.py -> /api/decode endpoint.
    Internally calls core/metadata_embedder, core/dwt_embedder, core/embedder.

Why this design:
    A user uploading a stego image doesn't know how it was embedded.
    By trying all methods in a sensible order (cheapest first), we hide
    that complexity from the user and the frontend.
"""

from pathlib import Path
from core.format_handler import classify, EmbeddingDomain


class DecodeResult:
    """
    Structured output from the universal decoder.

    Attributes:
        success        : True if a message was successfully extracted
        message        : the decoded message string (empty string if none)
        method_used    : which decoding path succeeded
        format_detected: the actual detected format string
        error          : human-readable error description if success=False
        warnings       : list of non-fatal warnings
    """

    def __init__(
        self,
        success         : bool,
        message         : str,
        method_used     : str,
        format_detected : str,
        error           : str = "",
        warnings        : list[str] | None = None,
    ):
        self.success         = success
        self.message         = message
        self.method_used     = method_used
        self.format_detected = format_detected
        self.error           = error
        self.warnings        = warnings or []

    def __str__(self):
        if self.success:
            return (
                f"[DECODE SUCCESS]\n"
                f"Method  : {self.method_used}\n"
                f"Format  : {self.format_detected}\n"
                f"Message : {self.message!r}\n"
                + (f"Warnings: {self.warnings}" if self.warnings else "")
            )
        return (
            f"[DECODE FAILED]\n"
            f"Method  : {self.method_used}\n"
            f"Format  : {self.format_detected}\n"
            f"Error   : {self.error}\n"
            + (f"Warnings: {self.warnings}" if self.warnings else "")
        )


def decode(image_path: str) -> DecodeResult:
    """
    Universal decoder: tries all known embedding methods in order
    until one succeeds or all are exhausted.

    Decoding order (cheapest first):
        1. Metadata   — reads file chunks, no pixel access
        2. DWT        — frequency domain, spatial formats only
        3. Spatial    — LSB replacement or LSB matching (identical read path)
        4. DCT        — JPEG images fall here; returns a helpful error message

    Args:
        image_path: path to the stego image (any supported format)

    Returns:
        DecodeResult — never raises, all errors are structured.
    """
    p        = Path(image_path)
    warnings = []

    if not p.exists():
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "none",
            format_detected = "unknown",
            error           = f"File not found: {image_path}",
        )

    try:
        info = classify(image_path)
    except Exception as e:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "none",
            format_detected = "unknown",
            error           = f"Format classification failed: {e}",
        )

    format_str = info.actual_format.value

    if info.extension_mismatch:
        warnings.append(
            f"Extension '{p.suffix}' does not match detected format "
            f"'{format_str}'. The file may have been renamed or converted."
        )

    if not info.is_supported:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "none",
            format_detected = format_str,
            error           = (
                f"Format '{format_str}' is not supported for decoding. "
                f"Supported: PNG, JPEG, TIFF, WebP."
            ),
            warnings        = warnings,
        )

    # Step 1: Metadata is cheapest — only reads file headers, no pixel work.
    result = _try_metadata(image_path, format_str, warnings)
    if result.success:
        return result

    # Steps 2 & 3: DWT and spatial LSB only apply to lossless (non-JPEG) images.
    # JPEG's lossy compression would have already destroyed any spatial LSB payload.
    if info.embedding_domain == EmbeddingDomain.SPATIAL:
        result = _try_dwt(image_path, format_str, warnings)
        if result.success:
            return result

    if info.embedding_domain == EmbeddingDomain.SPATIAL:
        return _decode_spatial_path(image_path, format_str, warnings)

    # Step 4: JPEG images reach here only after metadata failed.
    # If someone embedded via LSB on a JPEG, the output was actually a PNG —
    # the user needs to upload that PNG, not the original JPEG.
    if info.embedding_domain == EmbeddingDomain.DCT:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "none",
            format_detected = format_str,
            error           = (
                "No hidden message found in this JPEG. "
                "If you embedded using LSB Matching, LSB Replacement, or DWT, "
                "the output would have been a PNG file \u2014 please upload that PNG. "
                "If you embedded using Metadata, the EXIF data may have been "
                "stripped by a social media platform or image editor."
            ),
            warnings        = warnings,
        )

    return DecodeResult(
        success         = False,
        message         = "",
        method_used     = "none",
        format_detected = format_str,
        error           = f"No supported decoder found for format '{format_str}'.",
        warnings        = warnings,
    )


def _try_metadata(
    image_path: str, format_str: str, warnings: list[str]
) -> DecodeResult:
    """
    Attempt metadata decode. Returns success=False silently if no payload found.
    A ValueError from decode_metadata means "no payload", not an error.
    """
    try:
        from core.metadata_embedder import decode_metadata
        message = decode_metadata(image_path)
        return DecodeResult(
            success         = True,
            message         = message,
            method_used     = "metadata",
            format_detected = format_str,
            warnings        = warnings,
        )
    except ValueError:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "metadata",
            format_detected = format_str,
        )
    except Exception:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "metadata",
            format_detected = format_str,
        )


def _try_dwt(
    image_path: str, format_str: str, warnings: list[str]
) -> DecodeResult:
    """Attempt DWT decode. Returns success=False silently if no payload found."""
    try:
        from core.dwt_embedder import decode_dwt
        message = decode_dwt(image_path)
        return DecodeResult(
            success         = True,
            message         = message,
            method_used     = "dwt",
            format_detected = format_str,
            warnings        = warnings,
        )
    except (ValueError, UnicodeDecodeError):
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "dwt",
            format_detected = format_str,
        )
    except Exception:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "dwt",
            format_detected = format_str,
        )


def _decode_spatial_path(
    image_path: str, format_str: str, warnings: list[str]
) -> DecodeResult:
    """
    Decode using spatial LSB — the final attempt for spatial formats.
    Unlike _try_metadata and _try_dwt, this returns a meaningful error
    because there are no more methods to try after this one.
    """
    try:
        from core.embedder import decode as spatial_decode
        message = spatial_decode(image_path)
        return DecodeResult(
            success         = True,
            message         = message,
            method_used     = "spatial_lsb",
            format_detected = format_str,
            warnings        = warnings,
        )

    except ValueError as e:
        error_str = str(e)
        if "Terminator not found" in error_str:
            error_str = (
                "No hidden message found in this image. "
                "Tried metadata, DWT, and spatial LSB \u2014 all failed. "
                "The image may not contain a Scry-embedded message, "
                "or it was modified after embedding."
            )
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "spatial_lsb",
            format_detected = format_str,
            error           = error_str,
            warnings        = warnings,
        )

    except UnicodeDecodeError:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "spatial_lsb",
            format_detected = format_str,
            error           = (
                "No hidden message found in this image. "
                "Tried metadata, DWT, and spatial LSB \u2014 all failed. "
                "The image may have been modified or re-saved after embedding."
            ),
            warnings        = warnings,
        )

    except Exception as e:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "spatial_lsb",
            format_detected = format_str,
            error           = f"Decode failed unexpectedly: {e}",
            warnings        = warnings,
        )


def decode_with_format_hint(image_path: str, expected_format: str) -> DecodeResult:
    """
    Decode with an explicit format hint for cross-format mismatch diagnosis.
    Adds a warning if the detected format differs from what was expected.
    """
    result = decode(image_path)
    if result.format_detected.upper() != expected_format.upper():
        result.warnings.append(
            f"Format mismatch: you specified '{expected_format}' but the "
            f"image was detected as '{result.format_detected}'. "
            f"If the image was converted after encoding, the original "
            f"embedding method may no longer apply."
        )
    return result