"""
Universal decoder entry point for all supported formats.

This module is the single entry point for all decoding operations.
The caller never needs to know which embedding method was used —
the decoder detects it automatically from the format and embedded header.

Supported paths:
    Spatial (PNG, BMP, TIFF, lossless WebP) → core/embedder.py decode()
    DCT     (JPEG, lossy WebP)               → core/dct_embedder.py decode_dct()

Failure philosophy:
    Every failure mode must return a structured, human-readable error.
    No silent corruption. No raw exceptions reaching the caller.
    A user who uploads a re-saved JPEG receives a clear explanation,
    not garbage text or an unhandled exception.
"""

from pathlib import Path
from core.format_handler import classify, ImageFormat, CompressionType, EmbeddingDomain


class DecodeResult:
    """
    Structured output from the universal decoder.

    Attributes:
        success       : True if a message was successfully extracted
        message       : the decoded message string (empty string if none)
        method_used   : which decoding path was taken ("spatial" or "dct")
        format_detected: the actual detected format string
        error         : human-readable error description if success=False
        warnings      : list of non-fatal warnings (e.g. extension mismatch)
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
        self.success          = success
        self.message          = message
        self.method_used      = method_used
        self.format_detected  = format_detected
        self.error            = error
        self.warnings         = warnings or []

    def __str__(self):
        if self.success:
            return (
                f"[DECODE SUCCESS]\n"
                f"Method  : {self.method_used}\n"
                f"Format  : {self.format_detected}\n"
                f"Message : {self.message!r}\n"
                + (f"Warnings: {self.warnings}" if self.warnings else "")
            )
        else:
            return (
                f"[DECODE FAILED]\n"
                f"Method  : {self.method_used}\n"
                f"Format  : {self.format_detected}\n"
                f"Error   : {self.error}\n"
                + (f"Warnings: {self.warnings}" if self.warnings else "")
            )


def decode(image_path: str) -> DecodeResult:
    """
    Universal decoder — automatically selects the correct decoding path
    based on the detected image format.

    Routing logic:
        JPEG                → DCT decoder
        WebP lossy          → DCT decoder
        PNG                 → Spatial LSB decoder
        BMP                 → Spatial LSB decoder
        TIFF                → Spatial LSB decoder
        WebP lossless       → Spatial LSB decoder
        Anything else       → Structured error (unsupported format)

    Args:
        image_path: path to the stego image (any supported format)

    Returns:
        DecodeResult with success flag, message, and full diagnostic info.
        Never raises — all exceptions are caught and returned as
        structured errors in DecodeResult.
    """
    p = Path(image_path)
    warnings = []

    # File existence check
    if not p.exists():
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "none",
            format_detected = "unknown",
            error           = f"File not found: {image_path}",
        )

    # Classify the image — detect actual format from magic bytes
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

    # Warn about extension mismatches — they don't block decoding
    # but indicate the file may have been renamed or converted
    if info.extension_mismatch:
        warnings.append(
            f"Extension '{p.suffix}' does not match detected format "
            f"'{format_str}'. The file may have been renamed or converted."
        )

    # Warn about unsupported formats early
    if not info.is_supported:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "none",
            format_detected = format_str,
            error           = (
                f"Format '{format_str}' is not supported for decoding. "
                f"Supported formats: PNG, BMP, TIFF, JPEG, WebP."
            ),
            warnings        = warnings,
        )

    # ---------------------------------------------------------------------------
    # Route to the correct decoder based on embedding domain
    # ---------------------------------------------------------------------------

    if info.embedding_domain == EmbeddingDomain.DCT:
        return _decode_dct_path(image_path, format_str, warnings)

    elif info.embedding_domain == EmbeddingDomain.SPATIAL:
        return _decode_spatial_path(image_path, format_str, warnings)

    else:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "none",
            format_detected = format_str,
            error           = (
                f"No supported embedding domain found for format '{format_str}'. "
                f"This format cannot be decoded."
            ),
            warnings        = warnings,
        )


def _decode_spatial_path(
    image_path: str, format_str: str, warnings: list[str]
) -> DecodeResult:
    """
    Decode using the spatial LSB decoder from core/embedder.py.
    Used for: PNG, BMP, TIFF, lossless WebP.
    """
    try:
        from core.embedder import decode as spatial_decode
        message = spatial_decode(image_path)
        return DecodeResult(
            success         = True,
            message         = message,
            method_used     = "spatial",
            format_detected = format_str,
            warnings        = warnings,
        )

    except ValueError as e:
        # Structured errors from the spatial decoder
        error_str = str(e)

        # Provide additional context for common failure modes
        if "Terminator not found" in error_str:
            error_str += (
                " If this image was converted from JPEG or saved with "
                "lossy compression after encoding, the message cannot "
                "be recovered via spatial decoding."
            )

        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "spatial",
            format_detected = format_str,
            error           = error_str,
            warnings        = warnings,
        )

    except UnicodeDecodeError:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "spatial",
            format_detected = format_str,
            error           = (
                "Extracted bytes are not valid UTF-8. "
                "The image may have been modified after encoding, "
                "or does not contain a hidden message."
            ),
            warnings        = warnings,
        )

    except Exception as e:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "spatial",
            format_detected = format_str,
            error           = f"Spatial decode failed unexpectedly: {e}",
            warnings        = warnings,
        )


def _decode_dct_path(
    image_path: str, format_str: str, warnings: list[str]
) -> DecodeResult:
    """
    Decode using the DCT decoder from core/dct_embedder.py.
    Used for: JPEG, lossy WebP.
    """
    try:
        from core.dct_embedder import decode_dct
        message = decode_dct(image_path)
        return DecodeResult(
            success         = True,
            message         = message,
            method_used     = "dct",
            format_detected = format_str,
            warnings        = warnings,
        )

    except ValueError as e:
        error_str = str(e)

        # Enrich recompression failure message
        if "recompressed" in error_str.lower() or "terminator not found" in error_str.lower():
            error_str += (
                " Tip: DCT-embedded messages survive recompression only at "
                "the same quality setting used during embedding. "
                "Recompression at a different quality destroys the message."
            )

        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "dct",
            format_detected = format_str,
            error           = error_str,
            warnings        = warnings,
        )

    except UnicodeDecodeError:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "dct",
            format_detected = format_str,
            error           = (
                "Extracted bytes are not valid UTF-8. "
                "This JPEG image appears to have been recompressed after "
                "encoding — the hidden message could not be recovered."
            ),
            warnings        = warnings,
        )

    except Exception as e:
        return DecodeResult(
            success         = False,
            message         = "",
            method_used     = "dct",
            format_detected = format_str,
            error           = f"DCT decode failed unexpectedly: {e}",
            warnings        = warnings,
        )


def decode_with_format_hint(image_path: str, expected_format: str) -> DecodeResult:
    """
    Decode with an explicit format hint for cross-format mismatch diagnosis.

    If the detected format differs from the expected format, the result
    includes a warning explaining the mismatch and what format was
    used at encoding time.

    Args:
        image_path      : path to the stego image
        expected_format : format string the user believes was used
                          (e.g. "PNG", "JPEG")

    Returns:
        DecodeResult — same as decode() but with additional mismatch context.
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