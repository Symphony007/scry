# core/dct_embedder.py

import numpy as np
from PIL import Image
from pathlib import Path

METHOD_LSB = 0b0001
METHOD_DCT = 0b0010

FORMAT_CODES = {
    "PNG"  : 0b0001,
    "JPEG" : 0b0010,
    "BMP"  : 0b0011,
    "TIFF" : 0b0100,
    "WEBP" : 0b0101,
}
FORMAT_CODES_REVERSE = {v: k for k, v in FORMAT_CODES.items()}

TERMINATOR  = [0] * 16
BLOCK_SIZE  = 8

ZIGZAG_INDEX = [
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63,
]

MID_FREQ_POSITIONS = [
    (0, 3), (0, 4),
    (1, 2), (1, 3),
    (2, 1), (2, 2),
    (3, 0), (3, 1),
    (4, 0),
]

# Coefficients with |quantized value| < STABILITY_THRESHOLD are skipped
# in both embed and decode — they round unpredictably during recompression.
STABILITY_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def _build_header(method_id: int, format_code: int) -> list[int]:
    header_int = ((method_id & 0xF) << 12) | ((format_code & 0xF) << 8)
    return [(header_int >> i) & 1 for i in range(15, -1, -1)]


def _parse_header(bits: list[int]) -> tuple[int, int]:
    if len(bits) < 16:
        raise ValueError("Header too short — fewer than 16 bits available.")
    header_int = 0
    for b in bits[:16]:
        header_int = (header_int << 1) | b
    return (header_int >> 12) & 0xF, (header_int >> 8) & 0xF


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
        raise ValueError(f"Bit count {len(bits)} is not a multiple of 8.")
    ba = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        ba.append(byte)
    return ba.decode("utf-8")


# ---------------------------------------------------------------------------
# DCT helpers
# ---------------------------------------------------------------------------

def _dct_2d(block: np.ndarray) -> np.ndarray:
    from scipy.fft import dct
    tmp = dct(block.astype(np.float64), type=2, norm="ortho", axis=1)
    return dct(tmp, type=2, norm="ortho", axis=0)


def _idct_2d(block: np.ndarray) -> np.ndarray:
    from scipy.fft import idct
    tmp = idct(block.astype(np.float64), type=2, norm="ortho", axis=0)
    return idct(tmp, type=2, norm="ortho", axis=1)


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _q_value(qtable_zigzag: list[int], row: int, col: int) -> int:
    zz_idx = ZIGZAG_INDEX[row * 8 + col]
    return max(1, qtable_zigzag[zz_idx])


def _get_jpeg_qtables(image_path: str) -> tuple[dict | None, int]:
    try:
        with Image.open(image_path) as img:
            if not hasattr(img, "quantization") or not img.quantization:
                return None, 85
            qtables = dict(img.quantization)
            luma    = qtables.get(0, [])
            if not luma:
                return qtables, 85
            s = sum(luma)
            if   s <= 80 : quality = 97
            elif s <= 150: quality = 95
            elif s <= 250: quality = 92
            elif s <= 400: quality = 88
            elif s <= 580: quality = 85
            elif s <= 800: quality = 78
            elif s <= 1100: quality = 70
            else          : quality = 60
            return qtables, quality
    except Exception:
        return None, 85


def _default_qtables(quality: int) -> dict:
    base_luma = [
        16, 11, 12, 14, 12, 10, 16, 14,
        13, 14, 18, 17, 16, 19, 24, 40,
        26, 24, 22, 22, 24, 49, 35, 37,
        29, 40, 58, 51, 61, 60, 57, 51,
        56, 55, 64, 72, 92, 78, 64, 68,
        87, 69, 55, 56, 80,109, 81, 87,
        95, 98,103,104,103, 62, 77,113,
       121,112,100,120, 92,101,103, 99,
    ]
    base_chroma = [
        17, 18, 18, 24, 21, 24, 47, 26,
        26, 47, 99, 66, 56, 66, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ]
    q = max(1, min(95, quality))
    scale = (5000 // q) if q < 50 else (200 - 2 * q)

    def scale_table(base):
        return [max(1, min(255, (v * scale + 50) // 100)) for v in base]

    return {0: scale_table(base_luma), 1: scale_table(base_chroma)}


# ---------------------------------------------------------------------------
# Block splitting / reconstruction
# ---------------------------------------------------------------------------

def _get_channel_blocks(channel: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w   = channel.shape
    pad_h  = (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    pad_w  = (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")
    bh     = padded.shape[0] // BLOCK_SIZE
    bw     = padded.shape[1] // BLOCK_SIZE
    blocks = padded.reshape(bh, BLOCK_SIZE, bw, BLOCK_SIZE)
    blocks = blocks.transpose(0, 2, 1, 3)
    return blocks.astype(np.float64), pad_h, pad_w


def _reconstruct_channel(
    blocks: np.ndarray, orig_h: int, orig_w: int
) -> np.ndarray:
    bh, bw = blocks.shape[0], blocks.shape[1]
    ch = blocks.transpose(0, 2, 1, 3).reshape(bh * BLOCK_SIZE, bw * BLOCK_SIZE)
    return ch[:orig_h, :orig_w]


def _count_capacity(channel: np.ndarray) -> int:
    h, w = channel.shape
    bh   = (h + BLOCK_SIZE - 1) // BLOCK_SIZE
    bw   = (w + BLOCK_SIZE - 1) // BLOCK_SIZE
    return bh * bw * len(MID_FREQ_POSITIONS)


# ---------------------------------------------------------------------------
# Core embed
# ---------------------------------------------------------------------------

def embed_dct(image_path: str, message: str, output_path: str) -> dict:
    """
    Embed a UTF-8 message into a JPEG image using quantized DCT coefficients.

    Key design decisions:
        - We work directly in YCbCr space and save the YCbCr array as JPEG.
          This avoids a YCbCr → RGB → YCbCr round trip which loses precision
          and corrupts embedded bits.
        - Only coefficients with |quantized value| >= STABILITY_THRESHOLD
          are used. Near-zero coefficients round unpredictably on recompression.
        - Decoder skips the same positions — alignment is maintained exactly.
    """
    from core.format_handler import classify, ImageFormat, CompressionType

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    info = classify(image_path)
    if info.actual_format not in (ImageFormat.JPEG, ImageFormat.WEBP):
        raise ValueError(
            f"DCT embedder requires JPEG or lossy WebP. "
            f"Got: {info.actual_format.value}. "
            f"Use the spatial LSB embedder for PNG/BMP/TIFF."
        )
    if (info.actual_format == ImageFormat.WEBP and
            info.compression == CompressionType.LOSSLESS):
        raise ValueError(
            "DCT embedder cannot be used on lossless WebP. "
            "Use the spatial LSB embedder instead."
        )

    qtables, quality = _get_jpeg_qtables(image_path)
    if qtables is None:
        qtables = _default_qtables(85)
        quality = 85
        print("[DCT EMBED] No Q tables found — using standard Q85 tables.")

    luma_qtable = qtables.get(0, list(_default_qtables(quality)[0]))

    # Load directly as YCbCr — no RGB conversion
    ycbcr_arr   = np.array(Image.open(image_path).convert("YCbCr"), dtype=np.uint8)
    y_orig      = ycbcr_arr[:, :, 0].astype(np.float64)
    orig_h, orig_w = y_orig.shape

    format_code  = FORMAT_CODES.get(info.actual_format.value, FORMAT_CODES["JPEG"])
    header_bits  = _build_header(METHOD_DCT, format_code)
    msg_bits     = _text_to_bits(message)
    payload      = header_bits + msg_bits + TERMINATOR
    total_bits   = len(payload)

    capacity_bits = _count_capacity(y_orig)
    if total_bits > capacity_bits:
        raise ValueError(
            f"Message too large. Needs {total_bits} bits, "
            f"capacity is {capacity_bits} bits."
        )

    blocks, pad_h, pad_w = _get_channel_blocks(y_orig - 128.0)
    bh, bw = blocks.shape[0], blocks.shape[1]

    bit_index = 0
    done      = False

    for r in range(bh):
        if done:
            break
        for c in range(bw):
            if done:
                break

            dct_block = _dct_2d(blocks[r, c])

            for (row, col) in MID_FREQ_POSITIONS:
                if bit_index >= total_bits:
                    done = True
                    break

                Q       = _q_value(luma_qtable, row, col)
                q_coeff = int(round(dct_block[row, col] / Q))

                # Skip near-zero — rounds unpredictably on recompression
                if abs(q_coeff) < STABILITY_THRESHOLD:
                    continue

                # Modify LSB of stable quantized integer
                q_coeff = (q_coeff & ~1) | payload[bit_index]

                # Dequantize back to float domain
                dct_block[row, col] = float(q_coeff * Q)
                bit_index += 1

            blocks[r, c] = _idct_2d(dct_block)

    if bit_index < total_bits:
        raise ValueError(
            f"Not enough stable coefficients to embed message. "
            f"Needed {total_bits} bits, only found {bit_index} stable positions. "
            f"Try a shorter message or a more textured image."
        )

    y_modified = _reconstruct_channel(blocks + 128.0, orig_h, orig_w)
    y_modified  = np.clip(np.round(y_modified), 0, 255).astype(np.uint8)

    out_ycbcr = ycbcr_arr.copy()
    out_ycbcr[:, :, 0] = y_modified

    out_path = Path(output_path)
    if out_path.suffix.lower() not in (".jpg", ".jpeg"):
        out_path = out_path.with_suffix(".jpg")
        print(f"[DCT EMBED] Output changed to {out_path} (DCT requires JPEG).")

    # Save YCbCr array directly — no RGB conversion, no precision loss
    Image.fromarray(out_ycbcr, mode="YCbCr").save(
        str(out_path),
        format      = "JPEG",
        qtables     = qtables,
        subsampling = 0,
    )

    payload_pct = (total_bits / capacity_bits) * 100
    print(f"[DCT EMBED] Embedded {total_bits} bits ({payload_pct:.2f}% capacity).")
    print(f"[DCT EMBED] Q tables preserved. Est. quality: {quality}.")

    return {
        "bits_used"    : total_bits,
        "capacity_bits": capacity_bits,
        "payload_pct"  : payload_pct,
        "quality"      : quality,
    }


# ---------------------------------------------------------------------------
# Core decode
# ---------------------------------------------------------------------------

def decode_dct(image_path: str) -> str:
    """
    Extract a hidden message from a DCT-embedded JPEG image.

    Mirrors embed_dct exactly:
        - Load directly as YCbCr — no RGB conversion
        - Same Q tables, same positions, same stability skip logic
    """
    from core.format_handler import classify, ImageFormat

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    info = classify(image_path)
    if info.actual_format not in (ImageFormat.JPEG, ImageFormat.WEBP):
        raise ValueError(
            f"DCT decoder requires JPEG or WebP. Got: {info.actual_format.value}."
        )

    qtables, _ = _get_jpeg_qtables(image_path)
    if qtables is None:
        raise ValueError(
            "Could not read quantization tables from this image. "
            "The image may not be a standard JPEG, or the Q tables were removed."
        )

    luma_qtable = qtables.get(0, list(_default_qtables(85)[0]))

    # Load directly as YCbCr — must match embed path exactly
    y_arr = np.array(
        Image.open(image_path).convert("YCbCr")
    )[:, :, 0].astype(np.float64)

    blocks, _, _ = _get_channel_blocks(y_arr - 128.0)
    bh, bw = blocks.shape[0], blocks.shape[1]

    extracted_bits = []
    for r in range(bh):
        for c in range(bw):
            dct_block = _dct_2d(blocks[r, c])
            for (row, col) in MID_FREQ_POSITIONS:
                Q       = _q_value(luma_qtable, row, col)
                q_coeff = int(round(dct_block[row, col] / Q))

                # Skip same positions as embed — alignment is critical
                if abs(q_coeff) < STABILITY_THRESHOLD:
                    continue

                extracted_bits.append(q_coeff & 1)

    if len(extracted_bits) < 16:
        raise ValueError(
            "Not enough stable coefficients found. "
            "This image may not contain a hidden message, "
            "or was recompressed at a different quality after encoding."
        )

    try:
        method_id, format_code = _parse_header(extracted_bits[:16])
    except ValueError as e:
        raise ValueError(
            f"Header unreadable or corrupted. "
            f"This image may not contain a hidden message. Detail: {e}"
        )

    if method_id != METHOD_DCT:
        raise ValueError(
            f"Header method ID {method_id:#06b} does not match DCT "
            f"({METHOD_DCT:#06b}). Try the spatial LSB decoder instead."
        )

    encoded_format = FORMAT_CODES_REVERSE.get(format_code, "UNKNOWN")
    if encoded_format == "UNKNOWN":
        raise ValueError(
            f"Unrecognised format code {format_code:#06b} in header. "
            f"The header may have been damaged by recompression."
        )

    payload_bits = extracted_bits[16:]
    message_bits = None

    for byte_index in range(0, (len(payload_bits) // 8) - 1):
        bit_pos = byte_index * 8
        if payload_bits[bit_pos: bit_pos + 16] == TERMINATOR:
            message_bits = payload_bits[:bit_pos]
            break

    if message_bits is None:
        raise ValueError(
            f"Message terminator not found within image capacity. "
            f"Image was encoded as {encoded_format}. "
            f"If recompressed at a different quality or format-converted "
            f"after encoding, the message cannot be recovered."
        )

    if len(message_bits) == 0:
        return ""

    if len(message_bits) % 8 != 0:
        raise ValueError(
            f"Message bit count ({len(message_bits)}) is not a multiple of 8. "
            f"The message was likely damaged by recompression."
        )

    try:
        return _bits_to_text(message_bits)
    except UnicodeDecodeError:
        raise ValueError(
            f"Extracted bytes are not valid UTF-8. "
            f"This image (encoded as {encoded_format}) appears to have been "
            f"recompressed after encoding — the message could not be recovered."
        )