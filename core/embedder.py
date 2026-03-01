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
# Searched only at 8-bit boundaries in the decoder — never mid-byte.
TERMINATOR = [0] * 16


def embed(image_path: str, message: str, output_path: str) -> dict:
    """
    Embed a UTF-8 message into a lossless image using LSB replacement.

    Each bit of the message overwrites the LSB of sequential R, G, B
    channel values across pixels. Pixel values change by at most 1.
    A 16-bit zero terminator is appended after the message bits.

    Args:
        image_path  : path to the cover image (PNG, BMP, TIFF only)
        message     : the plaintext message to hide (any UTF-8 string)
        output_path : where to save the stego image

    Returns:
        A dict with keys:
            psnr          : float, quality of stego image vs original
            bits_used     : int, total bits written including terminator
            capacity      : dict from calculate_capacity()
            payload_pct   : float, percentage of capacity used

    Raises:
        ValueError: if the input format is lossy (JPEG etc.)
        ValueError: if the message is too large for the image
    """
    from pathlib import Path

    # Hard reject lossy input formats — LSB embedding cannot work on them
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

    # Flatten the image to a 1D channel stream: R0,G0,B0,R1,G1,B1,...
    flat = array.flatten()

    for i, bit in enumerate(payload):
        # Clear the LSB and set it to the message bit
        flat[i] = (flat[i] & 0xFE) | bit

    # Reshape back to original dimensions
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

    Reads LSBs sequentially from R, G, B channels. Searches for the
    16-bit zero terminator ONLY at byte-aligned positions (every 8 bits),
    never mid-byte. This prevents false terminator matches from space
    characters or other zero-adjacent bit patterns.

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

    # Extract all LSBs up to capacity
    bits = [(int(flat[i]) & 1) for i in range(max_bits)]

    # Search for the 16-bit zero terminator at byte-aligned boundaries only.
    # A byte boundary occurs every 8 bits. We check pairs of consecutive bytes
    # (i.e. 16 bits) starting at positions 0, 8, 16, 24, ...
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