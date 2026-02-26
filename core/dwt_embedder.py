"""
DWT (Discrete Wavelet Transform) Embedder

Embeds data in the frequency sub-bands of an image rather than
directly in pixel values. This makes the embedding more robust to
format changes and mild compression than spatial LSB methods.

How it works:
    1. Load image as RGB, operate on R channel only
    2. Apply one level of 2D Haar DWT — splits into four sub-bands:
           LL — low frequency (approximation) — not used, too visible
           LH — horizontal edges
           HL — vertical edges
           HH — diagonal edges (high frequency) — primary embedding band
    3. Modify LSBs of quantized HH coefficients to encode message bits
    4. Apply inverse DWT to reconstruct the modified R channel
    5. Recombine with original G/B channels and save as PNG

Why step=16 and stability threshold=4:
    Haar DWT coefficients involve sums/differences of pixel values.
    After IDWT and uint8 clamping, reloading and re-applying DWT
    produces coefficients with small rounding errors (~1-4 units).
    step=16 ensures a rounding error of 4 units only shifts q by 0.25,
    well below the 0.5 rounding threshold. Stability threshold=4
    discards coefficients too close to zero where sign instability
    could corrupt the bit stream.

Known limitations:
    - Capacity is lower than spatial LSB (~25% of LSB capacity)
    - PSNR typically 35-45 dB (mild quality loss accepted)
    - Does not survive heavy JPEG recompression (quality < 70)
    - Requires PyWavelets (pywt) library
"""

import numpy as np
from pathlib import Path
from PIL import Image

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

from core.utils import calculate_psnr

TERMINATOR          = [0] * 16
HAAR                = "haar"
STABILITY_THRESHOLD = 4

# 8-bit magic signature prepended to all DWT payloads.
# Allows decoder to verify this is a genuine DWT-embedded image
# and not an accidental match from a non-DWT image.
# Pattern: 10110101 = 0xB5
DWT_MAGIC = [1, 0, 1, 1, 0, 1, 0, 1]


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_pywt():
    if not PYWT_AVAILABLE:
        raise ImportError(
            "PyWavelets is required for DWT embedding. "
            "Install it with: pip install PyWavelets"
        )


# ---------------------------------------------------------------------------
# Text / bit conversion
# ---------------------------------------------------------------------------

def _text_to_bits(text: str) -> list[int]:
    raw  = text.encode("utf-8")
    bits = []
    for byte in raw:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_text(bits: list[int]) -> str:
    if len(bits) % 8 != 0:
        raise ValueError(f"Bit count {len(bits)} is not a multiple of 8.")
    ba = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        ba.append(byte)
    return ba.decode("utf-8")


# ---------------------------------------------------------------------------
# DWT helpers
# ---------------------------------------------------------------------------

def _dwt2(channel: np.ndarray) -> tuple:
    return pywt.dwt2(channel.astype(np.float64), HAAR)


def _idwt2(LL: np.ndarray, details: tuple) -> np.ndarray:
    return pywt.idwt2((LL, details), HAAR)


def _quantize(coeff: float, step: int) -> int:
    return int(np.round(coeff / step))


def _dequantize(q: int, step: int) -> float:
    return float(q * step)


def _count_capacity(hh: np.ndarray, step: int) -> int:
    count = 0
    for val in hh.flatten():
        if abs(_quantize(val, step)) >= STABILITY_THRESHOLD:
            count += 1
    return count


def get_dwt_capacity(arr: np.ndarray, step: int = 16) -> int:
    """
    Return the number of embeddable bits for a given image array.
    Used externally to size messages before calling embed_dwt.
    Subtracts magic header (8 bits) and terminator (16 bits) overhead.

    Args:
        arr  : image as numpy array (H, W, 3) uint8
        step : quantization step — must match the step used in embed_dwt

    Returns:
        Number of usable payload bits (0 if image is too small).
    """
    _check_pywt()
    R = arr[:, :, 0].astype(np.float64)
    _, (_, _, HH) = _dwt2(R)
    total_bits = _count_capacity(HH, step)
    # Subtract magic (8) + terminator (16) overhead
    return max(0, total_bits - 24)


# ---------------------------------------------------------------------------
# Core embed
# ---------------------------------------------------------------------------

def embed_dwt(
    image_path  : str,
    message     : str,
    output_path : str,
    step        : int = 16,
) -> dict:
    """
    Embed a UTF-8 message using DWT coefficient modification.

    Args:
        image_path  : path to cover image (any format)
        message     : UTF-8 message to hide
        output_path : path to save stego image (always saved as PNG)
        step        : quantization step — higher = more robust, lower capacity.
                      Default 16 is calibrated for uint8 PNG round-trip safety.

    Returns:
        dict with psnr, bits_used, capacity_bits, payload_pct, method, step, output_path

    Raises:
        ValueError  : if message exceeds capacity
        ImportError : if PyWavelets is not installed
    """
    _check_pywt()

    img      = Image.open(image_path).convert("RGB")
    original = np.array(img, dtype=np.uint8)

    R = original[:, :, 0].astype(np.float64)
    G = original[:, :, 1].copy()
    B = original[:, :, 2].copy()

    LL, (LH, HL, HH) = _dwt2(R)

    capacity_bits = _count_capacity(HH, step)

    message_bits = _text_to_bits(message)
    payload      = DWT_MAGIC + message_bits + TERMINATOR
    bits_needed  = len(payload)

    if bits_needed > capacity_bits:
        raise ValueError(
            f"Message too large for DWT embedding. "
            f"Needs {bits_needed} bits, stable HH capacity is {capacity_bits} bits. "
            f"Try a shorter message, a larger image, or reduce step size."
        )

    hh_flat   = HH.flatten().copy()
    bit_index = 0

    for i in range(len(hh_flat)):
        if bit_index >= bits_needed:
            break
        q = _quantize(hh_flat[i], step)
        if abs(q) < STABILITY_THRESHOLD:
            continue
        sign       = 1 if q >= 0 else -1
        abs_q      = abs(q)
        abs_q      = (abs_q & ~1) | payload[bit_index]
        q          = sign * abs_q
        hh_flat[i] = _dequantize(q, step)
        bit_index += 1

    if bit_index < bits_needed:
        raise ValueError(
            f"Not enough stable coefficients. "
            f"Embedded {bit_index} of {bits_needed} bits."
        )

    HH_modified = hh_flat.reshape(HH.shape)
    R_modified  = _idwt2(LL, (LH, HL, HH_modified))

    h, w       = R.shape
    R_modified = np.clip(np.round(R_modified[:h, :w]), 0, 255).astype(np.uint8)

    stego    = np.stack([R_modified, G, B], axis=2)
    out_path = Path(output_path).with_suffix(".png")
    Image.fromarray(stego).save(str(out_path), format="PNG")

    psnr        = calculate_psnr(original, stego)
    payload_pct = (bits_needed / capacity_bits) * 100

    print(f"[DWT EMBED] Embedded {bits_needed} bits ({payload_pct:.2f}% of HH capacity).")
    print(f"[DWT EMBED] PSNR: {psnr:.2f} dB  |  Step: {step}")

    return {
        "psnr"         : psnr,
        "bits_used"    : bits_needed,
        "capacity_bits": capacity_bits,
        "payload_pct"  : payload_pct,
        "method"       : "dwt",
        "step"         : step,
        "output_path"  : str(out_path),
    }


# ---------------------------------------------------------------------------
# Core decode
# ---------------------------------------------------------------------------

def decode_dwt(image_path: str, step: int = 16) -> str:
    """
    Extract a hidden message from a DWT-embedded image.

    Must use the same step size as was used during embedding.
    Default step=16 matches the embed default.

    Args:
        image_path : path to the stego image (PNG output from embed_dwt)
        step       : quantization step — must match embedding step

    Returns:
        Decoded message string.

    Raises:
        ValueError         : if terminator not found or message is corrupt
        UnicodeDecodeError : if extracted bytes are not valid UTF-8
    """
    _check_pywt()

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    R = arr[:, :, 0].astype(np.float64)

    _, (_, _, HH) = _dwt2(R)

    extracted = []
    for val in HH.flatten():
        q = _quantize(val, step)
        if abs(q) < STABILITY_THRESHOLD:
            continue
        extracted.append(abs(q) & 1)

    if len(extracted) < 8:
        raise ValueError(
            "Not enough stable coefficients to decode. "
            "This image may not contain a DWT-embedded message."
        )

    # Verify magic signature — rejects accidental matches from non-DWT images
    if extracted[:8] != DWT_MAGIC:
        raise ValueError(
            "DWT magic signature not found. "
            "This image was not embedded with the DWT method."
        )

    # Strip magic before searching for terminator
    extracted = extracted[8:]

    if len(extracted) < 16:
        raise ValueError(
            "Not enough data after magic signature."
        )

    message_bits = None
    for byte_index in range(0, (len(extracted) // 8) - 1):
        bit_pos = byte_index * 8
        if extracted[bit_pos: bit_pos + 16] == TERMINATOR:
            message_bits = extracted[:bit_pos]
            break

    if message_bits is None:
        raise ValueError(
            "Message terminator not found. "
            "This image may not contain a DWT-embedded message, "
            "or the step size does not match embedding."
        )

    if len(message_bits) == 0:
        return ""

    if len(message_bits) % 8 != 0:
        raise ValueError(
            f"Message bit count ({len(message_bits)}) is not a multiple of 8. "
            "The message may have been corrupted."
        )

    try:
        return _bits_to_text(message_bits)
    except UnicodeDecodeError:
        raise ValueError(
            "Extracted bytes are not valid UTF-8. "
            "The step size may not match, or the image was modified after embedding."
        )