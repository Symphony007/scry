# core/lsb_matching_embedder.py

"""
LSB Matching Embedder

Unlike LSB replacement which always sets the LSB directly (creating a
detectable statistical fingerprint), LSB matching achieves the target
LSB by randomly incrementing or decrementing the pixel value by 1.

This eliminates histogram combing entirely — even/odd pair counts
remain naturally distributed because changes go in both directions
randomly. Chi-square and histogram detectors score near zero on
LSB-matched images even at large payloads.

Capacity is identical to LSB replacement — 1 bit per channel.
PSNR is essentially identical — pixel changes are still at most 1.
The only cost is that embedding is slightly slower due to RNG calls.

Supported formats: PNG, BMP, TIFF (lossless spatial domain only).
JPEG is rejected — same rule as LSB replacement.
"""

import numpy as np
from pathlib import Path
from core.utils import (
    load_image,
    save_image,
    text_to_bits,
    calculate_capacity,
    calculate_psnr,
)

TERMINATOR = [0] * 16


def embed_matching(
    image_path  : str,
    message     : str,
    output_path : str,
    seed        : int | None = None,
) -> dict:
    """
    Embed a UTF-8 message using LSB Matching.

    For each bit to embed:
        - If the pixel's current LSB already matches the message bit,
          leave it alone.
        - If it doesn't match, randomly add or subtract 1.
          If the chosen direction would go out of [0, 255] bounds,
          use the other direction instead — this guarantees the flip
          always succeeds without corrupting the bit stream.

    Args:
        image_path  : path to cover image (PNG, BMP, TIFF only)
        message     : UTF-8 message to hide
        output_path : path to save stego image
        seed        : optional RNG seed for reproducibility in tests

    Returns:
        dict with psnr, bits_used, capacity, payload_pct, method

    Raises:
        ValueError: if format is lossy or message is too large
    """
    suffix = Path(image_path).suffix.lower()
    lossy  = {'.jpg', '.jpeg', '.webp'}
    if suffix in lossy:
        raise ValueError(
            f"LSB Matching requires a lossless format. "
            f"Got '{suffix}'. Use PNG, BMP, or TIFF."
        )

    original, _ = load_image(image_path)
    array       = original.copy()
    capacity    = calculate_capacity(array)

    message_bits = text_to_bits(message)
    payload      = message_bits + TERMINATOR
    bits_needed  = len(payload)

    if bits_needed > capacity["total_bits"]:
        raise ValueError(
            f"Message too large. "
            f"Needs {bits_needed} bits, image holds {capacity['total_bits']} bits."
        )

    rng  = np.random.default_rng(seed)
    flat = array.flatten().astype(np.int16)

    for i, bit in enumerate(payload):
        current_lsb = int(flat[i]) & 1

        if current_lsb == bit:
            # LSB already correct — leave pixel unchanged
            continue

        # LSB needs to flip — randomly pick +1 or -1
        delta = 1 if rng.integers(0, 2) == 0 else -1

        # If chosen delta would go out of bounds, use the other direction
        if flat[i] + delta < 0 or flat[i] + delta > 255:
            delta = -delta

        flat[i] = flat[i] + delta

    stego_array = flat.astype(np.uint8).reshape(array.shape)
    psnr        = calculate_psnr(original, stego_array)
    save_image(stego_array, output_path)

    payload_pct = (bits_needed / capacity["total_bits"]) * 100

    print(f"[LSB MATCH] Embedded {bits_needed} bits ({payload_pct:.2f}% capacity).")
    print(f"[LSB MATCH] PSNR: {psnr:.2f} dB")

    return {
        "psnr"       : psnr,
        "bits_used"  : bits_needed,
        "capacity"   : capacity,
        "payload_pct": payload_pct,
        "method"     : "lsb_matching",
    }


def decode_matching(image_path: str) -> str:
    """
    Decode a message embedded with LSB Matching.

    The decode path is identical to LSB replacement — both methods
    produce the same LSB pattern in the stego image. The difference
    is only in how the bits were written, not how they are read.

    Args:
        image_path: path to the stego image

    Returns:
        Decoded message string.

    Raises:
        ValueError: if terminator not found
        UnicodeDecodeError: if extracted bytes are not valid UTF-8
    """
    from core.embedder import decode as lsb_decode
    return lsb_decode(image_path)