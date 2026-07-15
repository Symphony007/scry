"""
core/embedder.py — LSB Replacement embedder and decoder

Purpose:
    Hides a UTF-8 message inside a lossless image by overwriting the
    Least Significant Bit (LSB) of sequential R, G, B channel values.

Inputs:
    - A path to a lossless cover image (PNG, TIFF)
    - A UTF-8 message string
    - An output path for the stego image

Outputs:
    - The stego image saved to output_path
    - A dict with: PSNR (quality), bits used, capacity, and payload %

How it fits in:
    Called by web/app.py when method="lsb_replacement".
    The decoder (decode()) is also called by core/decoder.py as
    the final fallback in the spatial decode chain.

Why LSB replacement:
    Each pixel value only changes by 0 or 1 — the change is visually
    invisible. A 1000x1000 RGB image holds 375,000 characters.
    Trade-off: it creates a detectable statistical signature (chi-square).
"""

import numpy as np
from core.utils import (
    load_image,
    save_image,
    text_to_bits,
    bits_to_text,
    calculate_capacity,
    calculate_psnr,
    LOSSY_FORMATS,
)

# 16-bit zero terminator appended after every message.
# Searched only at byte-aligned positions (every 8 bits) during decoding,
# never mid-byte. This prevents false matches from space characters (0x20)
# or other byte patterns that contain zero bits.
TERMINATOR = [0] * 16


def embed(image_path: str, message: str, output_path: str) -> dict:
    """
    Embed a UTF-8 message into a lossless image using LSB replacement.

    Each bit of the message overwrites the LSB of sequential R, G, B
    channel values across pixels. Pixel values change by at most 1.
    A 16-bit zero terminator is appended after the message bits.

    Args:
        image_path  : path to the cover image (PNG, TIFF only — not JPEG)
        message     : the plaintext message to hide (any UTF-8 string)
        output_path : where to save the stego image

    Returns:
        A dict with: psnr, bits_used, capacity, payload_pct

    Raises:
        ValueError: if the input format is lossy (JPEG etc.)
        ValueError: if the message is too large for the image
    """
    from pathlib import Path

    suffix = Path(image_path).suffix.lower()
    if suffix in LOSSY_FORMATS:
        raise ValueError(
            f"Cannot embed into a lossy format ('{suffix}'). "
            f"Use a PNG, TIFF, or lossless WebP file as the cover image."
        )

    original, _ = load_image(image_path)
    array = original.copy()

    capacity = calculate_capacity(array)
    message_bits = text_to_bits(message)
    payload = message_bits + TERMINATOR
    bits_needed = len(payload)

    if bits_needed > capacity["total_bits"]:
        raise ValueError(
            f"Message too large. "
            f"Needs {bits_needed} bits, image holds {capacity['total_bits']} bits "
            f"({capacity['usable_bytes']} usable bytes)."
        )

    # Flatten to a 1D stream: R0,G0,B0,R1,G1,B1,...
    # This lets us write bits sequentially across all channels and pixels.
    flat = array.flatten()

    for i, bit in enumerate(payload):
        # Clear the LSB (& 0xFE) and replace it with the message bit (| bit)
        flat[i] = (flat[i] & 0xFE) | bit

    stego_array = flat.reshape(array.shape)

    psnr = calculate_psnr(original, stego_array)
    save_image(stego_array, output_path)

    payload_pct = (bits_needed / capacity["total_bits"]) * 100

    print(f"[EMBED] Message embedded successfully.")
    print(f"[EMBED] PSNR: {psnr:.2f} dB")
    print(f"[EMBED] Payload: {payload_pct:.2f}% of capacity")

    return {
        "psnr": psnr,
        "bits_used": bits_needed,
        "capacity": capacity,
        "payload_pct": payload_pct,
    }


def decode(image_path: str) -> str:
    """
    Extract a hidden UTF-8 message from a stego image.

    Reads LSBs sequentially from R, G, B channels and searches for the
    16-bit zero terminator at byte-aligned positions only (every 8 bits).

    Why byte-aligned only: searching mid-byte would cause false terminator
    matches. For example, ASCII space (0x20 = 00100000) contains 6 zero
    bits that could combine with bits from adjacent bytes to fake a
    16-bit zero sequence.

    Args:
        image_path : path to the stego image

    Returns:
        The decoded message string.

    Raises:
        ValueError: if no terminator is found within image capacity
        UnicodeDecodeError: if the extracted bytes are not valid UTF-8
    """
    array, _ = load_image(image_path)
    flat = array.flatten()

    capacity = calculate_capacity(array)
    max_bits = capacity["total_bits"]

    bits = [(int(flat[i]) & 1) for i in range(max_bits)]

    message_bits = None
    for byte_index in range(0, (max_bits // 8) - 1):
        bit_pos = byte_index * 8
        window = bits[bit_pos: bit_pos + 16]
        if window == TERMINATOR:
            message_bits = bits[:bit_pos]
            break

    if message_bits is None:
        raise ValueError(
            "Terminator not found within image capacity. "
            "This image may not contain a hidden message, "
            "or it may have been altered after embedding."
        )

    if len(message_bits) == 0:
        return ""

    return bits_to_text(message_bits)