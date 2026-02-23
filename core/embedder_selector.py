"""
Adaptive Embedder Selector

Given a user-stated priority, selects and invokes the appropriate
embedding method. Never silently falls back — always tells the caller
which method was selected and why.

Available methods:
    lsb_replacement — maximum capacity, detectable by statistical analysis
    lsb_matching    — same capacity, eliminates histogram/chi-square fingerprint
    metadata        — zero pixel change, detectable by file size
    dwt             — format-flexible, survives some compression
    dct             — JPEG coefficient domain, survives same-quality recompression
"""

from dataclasses import dataclass
from pathlib import Path
from core.format_handler import classify, EmbeddingDomain


# ---------------------------------------------------------------------------
# Priority options the user can specify
# ---------------------------------------------------------------------------

class Priority:
    UNDETECTABILITY = "undetectability"   # LSB Matching
    CAPACITY        = "capacity"          # LSB Replacement
    JPEG_SURVIVAL   = "jpeg_survival"     # DCT / F5
    FORMAT_FLEXIBLE = "format_flexible"   # DWT
    PIXEL_PRESERVE  = "pixel_preserve"    # Metadata


ALL_PRIORITIES = [
    Priority.UNDETECTABILITY,
    Priority.CAPACITY,
    Priority.JPEG_SURVIVAL,
    Priority.FORMAT_FLEXIBLE,
    Priority.PIXEL_PRESERVE,
]


# ---------------------------------------------------------------------------
# Selection result
# ---------------------------------------------------------------------------

@dataclass
class SelectionResult:
    """
    Result of the embedder selection process.

    Attributes:
        method          : the method name that was selected
        priority        : the priority that was requested
        reasoning       : plain-English explanation of why this method was chosen
        embed_result    : the dict returned by the embedder (after embedding)
        warnings        : list of non-fatal warnings

    Note on embed_result shape:
        Spatial embedders (lsb_replacement, lsb_matching, dwt) return:
            {"psnr", "bits_used", "capacity"/"capacity_bits", "payload_pct", "method"}
        Metadata embedder returns:
            {"method", "bits_used", "pixel_delta", "format"}  — no "psnr" key
        DCT embedder returns:
            {"bits_used", "capacity_bits", "payload_pct", "quality"}
        Always use .get() with a default when accessing optional keys.
    """
    method       : str
    priority     : str
    reasoning    : str
    embed_result : dict
    warnings     : list[str]

    def __str__(self):
        lines = [
            f"Method   : {self.method}",
            f"Priority : {self.priority}",
            f"Reason   : {self.reasoning}",
            f"PSNR     : {self.embed_result.get('psnr', 'N/A')}",
            f"Payload  : {self.embed_result.get('payload_pct', 0):.2f}%",
        ]
        if self.warnings:
            lines.append(f"Warnings : {'; '.join(self.warnings)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core selector
# ---------------------------------------------------------------------------

def select_and_embed(
    image_path  : str,
    message     : str,
    output_path : str,
    priority    : str = Priority.UNDETECTABILITY,
) -> SelectionResult:
    """
    Select the best embedding method for the given priority and embed.

    Args:
        image_path  : path to cover image
        message     : UTF-8 message to hide
        output_path : path to save stego image
        priority    : one of the Priority constants

    Returns:
        SelectionResult with method, reasoning, and embed output.

    Raises:
        ValueError : if priority is invalid or format is unsupported
    """
    if priority not in ALL_PRIORITIES:
        raise ValueError(
            f"Unknown priority '{priority}'. "
            f"Valid options: {ALL_PRIORITIES}"
        )

    warnings = []

    # Classify the image format
    info = classify(image_path)

    if not info.is_supported:
        raise ValueError(
            f"Format '{info.actual_format.value}' is not supported for embedding."
        )

    # Warn about extension mismatches
    if info.extension_mismatch:
        warnings.append(
            f"Extension mismatch detected — "
            f"actual format is {info.actual_format.value}."
        )

    # ---------------------------------------------------------------------------
    # Method selection logic
    # ---------------------------------------------------------------------------

    if info.embedding_domain == EmbeddingDomain.DCT:
        # JPEG / lossy WebP input
        if priority in (Priority.JPEG_SURVIVAL, Priority.FORMAT_FLEXIBLE):
            method    = "dct"
            reasoning = (
                "DCT embedding selected — image is JPEG/lossy WebP. "
                "Spatial LSB methods cannot survive JPEG compression."
            )

        elif priority == Priority.PIXEL_PRESERVE:
            # Metadata handles JPEG natively via EXIF — no conversion needed
            method    = "metadata"
            reasoning = (
                "Metadata embedding selected — JPEG input handled natively "
                "via EXIF UserComment. No format conversion required. "
                "Zero pixel modification."
            )

        else:
            # UNDETECTABILITY or CAPACITY on a JPEG — convert to PNG first
            method    = _spatial_method_for_priority(priority)
            reasoning = (
                f"JPEG input detected. Converting to PNG and using "
                f"{method} — spatial LSB methods require a lossless format."
            )
            warnings.append(
                "JPEG converted to PNG. Output will be a PNG file."
            )
            image_path  = _convert_jpeg_to_png(image_path)
            output_path = str(Path(output_path).with_suffix(".png"))

    else:
        # Lossless spatial domain (PNG, BMP, TIFF, lossless WebP)
        method    = _spatial_method_for_priority(priority)
        reasoning = _reasoning_for_method(method, priority)

    # ---------------------------------------------------------------------------
    # Invoke the selected embedder
    # ---------------------------------------------------------------------------

    embed_result = _invoke_embedder(method, image_path, message, output_path)

    return SelectionResult(
        method       = method,
        priority     = priority,
        reasoning    = reasoning,
        embed_result = embed_result,
        warnings     = warnings,
    )


def _spatial_method_for_priority(priority: str) -> str:
    """Map a priority to a spatial embedding method."""
    if priority == Priority.UNDETECTABILITY:
        return "lsb_matching"
    elif priority == Priority.CAPACITY:
        return "lsb_replacement"
    elif priority == Priority.PIXEL_PRESERVE:
        return "metadata"
    elif priority == Priority.FORMAT_FLEXIBLE:
        return "dwt"
    elif priority == Priority.JPEG_SURVIVAL:
        return "dct"
    return "lsb_matching"


def _reasoning_for_method(method: str, priority: str) -> str:
    reasons = {
        "lsb_matching": (
            "LSB Matching selected — eliminates histogram combing and "
            "chi-square fingerprint. Statistical detectors score near zero. "
            "Recommended for any scenario where detection resistance matters."
        ),
        "lsb_replacement": (
            "LSB Replacement selected — maximum capacity, fastest embedding. "
            "Detectable by chi-square and histogram analysis. "
            "Use only when capacity is the priority and detection is not a concern."
        ),
        "metadata": (
            "Metadata embedder selected — zero pixel modification. "
            "File size increases detectably. Not suitable when file size is analysed. "
            "Note: stripped by most social media platforms."
        ),
        "dwt": (
            "DWT embedder selected — embeds in frequency sub-bands. "
            "More robust to format changes than spatial LSB methods."
        ),
        "dct": (
            "DCT embedder selected — embeds in JPEG coefficient domain. "
            "Survives recompression at the same quality setting."
        ),
    }
    return reasons.get(method, f"Method '{method}' selected for priority '{priority}'.")


def _convert_jpeg_to_png(image_path: str) -> str:
    """Convert a JPEG to a temporary PNG for spatial embedding."""
    import tempfile
    import os
    from PIL import Image

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.open(image_path).convert("RGB").save(tmp_path, format="PNG")
    return tmp_path


def _invoke_embedder(
    method      : str,
    image_path  : str,
    message     : str,
    output_path : str,
) -> dict:
    """
    Invoke the correct embedder for the selected method.

    Return value shapes differ per method — see SelectionResult docstring.
    """
    if method == "lsb_replacement":
        from core.embedder import embed
        return embed(image_path, message, output_path)

    elif method == "lsb_matching":
        from core.lsb_matching_embedder import embed_matching
        return embed_matching(image_path, message, output_path)

    elif method == "dct":
        from core.dct_embedder import embed_dct
        return embed_dct(image_path, message, output_path)

    elif method == "metadata":
        from core.metadata_embedder import embed_metadata
        return embed_metadata(image_path, message, output_path)

    elif method == "dwt":
        from core.dwt_embedder import embed_dwt
        return embed_dwt(image_path, message, output_path)

    else:
        raise ValueError(f"Unknown embedding method '{method}'.")