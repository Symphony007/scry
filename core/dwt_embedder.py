
"""
DWT (Discrete Wavelet Transform) Embedder

Embeds data in the frequency sub-bands of an image rather than
directly in pixel values. This makes the embedding more robust to
format changes and mild compression than spatial LSB methods.

How it works:
    1. Convert image to YCbCr, operate on Y (luma) channel only
    2. Apply one level of 2D Haar DWT — splits image into four
       sub-bands:
           LL — low frequency (approximation) — not used, too visible
           LH — horizontal edges
           HL — vertical edges
           HH — diagonal edges (high frequency) — primary embedding band
    3. Modify LSBs of quantized HH coefficients to encode message bits
    4. Apply inverse DWT to reconstruct the modified image
    5. Recombine with original Cb/Cr channels and convert back to RGB

Why HH sub-band:
    The HH sub-band contains high-frequency diagonal detail — the
    least perceptually significant part of the image. Changes here
    are harder to see and harder to detect statistically than
    changes in the spatial domain.

Why more robust than spatial LSB:
    Frequency domain changes spread across multiple pixels when
    transformed back to spatial domain. A small compression or
    resize operation affects spatial pixels directly but disturbs
    frequency coefficients more predictably.

Known limitations:
    - Capacity is lower than spatial LSB (~25% of LSB capacity)
      because only the HH sub-band is used
    - Mild quality loss is accepted — PSNR typically 38-45 dB
      vs 50-55 dB for spatial LSB
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

TERMINATOR  = [0] * 16
HAAR        = "haar"


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
    raw = text.encode("utf-8")
    bits = []
    for byte in raw:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_text(bits: list[int]) -> str:
    if len(bits) % 8 != 0:
        raise ValueError(
            f"Bit count {len(bits)} is not a multiple of 8."
        )
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
    """Apply one level of 2D Haar DWT. Returns (LL, (LH, HL, HH))."""
    return pywt.dwt2(channel.astype(np.float64), HAAR)


def _idwt2(LL: np.ndarray, details: tuple) -> np.ndarray:
    """Apply inverse 2D Haar DWT."""
    return pywt.idwt2((LL, details), HAAR)


def _quantize(coeff: float, step: int = 4) -> int:
    """Quantize a DWT coefficient to an integer for LSB manipulation."""
    return int(np.round(coeff / step))


def _dequantize(q: int, step: int = 4) -> float:
    """Dequantize back to float domain."""
    return float(q * step)


def _count_capacity(hh: np.ndarray, step: int = 4) -> int:
    """
    Count how many stable HH coefficients are available for embedding.
    Coefficients with |quantized value| < 2 are skipped — they round
    unpredictably and corrupt the bit stream.
    """
    count = 0
    flat  = hh.flatten()
    for val in flat:
        q = _quantize(val, step)
        if abs(q) >= 2:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Core embed
# ---------------------------------------------------------------------------

def embed_dwt(
    image_path  : str,
    message     : str,
    output_path : str,
    step        : int = 4,
) -> dict:
    _check_pywt()

    img      = Image.open(image_path).convert("RGB")
    original = np.array(img, dtype=np.uint8)

    # Work directly on R channel — no YCbCr conversion roundtrip
    R = original[:, :, 0].astype(np.float64)
    G = original[:, :, 1].copy()
    B = original[:, :, 2].copy()

    LL, (LH, HL, HH) = _dwt2(R)

    capacity_bits = _count_capacity(HH, step)

    message_bits = _text_to_bits(message)
    payload      = message_bits + TERMINATOR
    bits_needed  = len(payload)

    if bits_needed > capacity_bits:
        raise ValueError(
            f"Message too large for DWT embedding. "
            f"Needs {bits_needed} bits, "
            f"stable HH capacity is {capacity_bits} bits. "
            f"Try a shorter message, a larger image, or reduce step size."
        )

    hh_flat   = HH.flatten().copy()
    bit_index = 0

    for i in range(len(hh_flat)):
        if bit_index >= bits_needed:
            break
        q = _quantize(hh_flat[i], step)
        if abs(q) < 2:
            continue
        sign  = 1 if q >= 0 else -1
        abs_q = abs(q)
        abs_q = (abs_q & ~1) | payload[bit_index]
        q     = sign * abs_q
        hh_flat[i] = _dequantize(q, step)
        bit_index += 1

    if bit_index < bits_needed:
        raise ValueError(
            f"Not enough stable coefficients. "
            f"Embedded {bit_index} of {bits_needed} bits."
        )

    HH_modified = hh_flat.reshape(HH.shape)
    R_modified  = _idwt2(LL, (LH, HL, HH_modified))

    h, w = R.shape
    R_modified = np.clip(np.round(R_modified[:h, :w]), 0, 255).astype(np.uint8)

    # Reconstruct RGB with modified R channel
    stego = np.stack([R_modified, G, B], axis=2)

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

def decode_dwt(image_path: str, step: int = 4) -> str:
    _check_pywt()

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)

    # Read same R channel — must match embed path exactly
    R = arr[:, :, 0].astype(np.float64)

    _, (_, _, HH) = _dwt2(R)

    extracted = []
    hh_flat   = HH.flatten()

    for val in hh_flat:
        q = _quantize(val, step)
        if abs(q) < 2:
            continue
        extracted.append(abs(q) & 1)

    if len(extracted) < 16:
        raise ValueError(
            "Not enough stable coefficients to decode. "
            "This image may not contain a DWT-embedded message."
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
            "or the step size used during decoding does not match embedding."
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