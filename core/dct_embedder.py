import numpy as np
from PIL import Image
from pathlib import Path

# ---------------------------------------------------------------------------
# Embedding Header Format (16 bits)
# Bits 0-3  : Method ID  (0001=LSB spatial, 0010=DCT)
# Bits 4-7  : Format code at encoding time
# Bits 8-15 : Reserved (zeros)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Zigzag scan order for an 8x8 DCT block.
# ZIGZAG_INDEX[row * 8 + col] = zigzag position of (row, col).
# Pillow stores quantization tables in zigzag order, so we use this
# to look up the Q value for any (row, col) position.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Mid-frequency DCT positions for embedding.
# These are low-to-mid frequency positions (small zigzag index = smaller Q
# values = more stable round-trips). DC coefficient (0,0) is never touched.
# ---------------------------------------------------------------------------
MID_FREQ_POSITIONS = [
    (0, 3), (0, 4),
    (1, 2), (1, 3),
    (2, 1), (2, 2),
    (3, 0), (3, 1),
    (4, 0),
]


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def _build_header(method_id: int, format_code: int) -> list[int]:
    """Build 16-bit header as a list of bits (MSB first)."""
    header_int = ((method_id & 0xF) << 12) | ((format_code & 0xF) << 8)
    return [(header_int >> i) & 1 for i in range(15, -1, -1)]


def _parse_header(bits: list[int]) -> tuple[int, int]:
    """Parse 16-bit header. Returns (method_id, format_code)."""
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
# DCT helpers (scipy)
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
# Quantization table helpers
# ---------------------------------------------------------------------------

def _q_value(qtable_zigzag: list[int], row: int, col: int) -> int:
    """Return the quantization step for DCT position (row, col)."""
    zz_idx = ZIGZAG_INDEX[row * 8 + col]
    return max(1, qtable_zigzag[zz_idx])


def _get_jpeg_qtables(image_path: str) -> tuple[dict | None, int]:
    """
    Extract quantization tables and quality estimate from a JPEG.

    Returns:
        (qtables, quality)
        qtables : dict {0: [64 ints zigzag], 1: [64 ints zigzag]}
                  in the exact format Pillow's qtables= parameter expects.
                  None if not a JPEG or tables unavailable.
        quality : estimated quality int (fallback 85)
    """
    try:
        with Image.open(image_path) as img:
            if not hasattr(img, "quantization") or not img.quantization:
                return None, 85
            qtables = dict(img.quantization)
            luma    = qtables.get(0, [])
            if not luma:
                return qtables, 85
            # Estimate quality from sum of luma Q table values.
            # Standard JPEG luma table at quality 50 sums to ~580 (8-bit).
            # Quality scales inversely.
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
    """
    Generate standard JPEG quantization tables for a given quality.
    Used as fallback when the source has no embedded Q tables.
    Quality must be in [1, 95].
    """
    # Standard JPEG luma table (zigzag order)
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
    # Standard JPEG chroma table (zigzag order)
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
    """Split channel into 8x8 blocks. Returns (blocks, pad_h, pad_w)."""
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
    """Reconstruct channel from 8x8 blocks, strip padding."""
    bh, bw = blocks.shape[0], blocks.shape[1]
    ch = blocks.transpose(0, 2, 1, 3).reshape(bh * BLOCK_SIZE, bw * BLOCK_SIZE)
    return ch[:orig_h, :orig_w]


def _count_capacity(channel: np.ndarray) -> int:
    h, w = channel.shape
    bh   = (h + BLOCK_SIZE - 1) // BLOCK_SIZE
    bw   = (w + BLOCK_SIZE - 1) // BLOCK_SIZE
    return bh * bw * len(MID_FREQ_POSITIONS)


# ---------------------------------------------------------------------------
# Core embed / decode
# ---------------------------------------------------------------------------

def embed_dct(image_path: str, message: str, output_path: str) -> dict:
    """
    Embed a UTF-8 message into a JPEG image using quantized DCT coefficients.

    How it survives JPEG re-save:
        Instead of modifying raw float DCT coefficients (which Pillow
        overwrites during save), we:
          1. Read the source's own quantization tables
          2. DCT each 8x8 block
          3. Quantize with the source Q table → integer coefficients
          4. Modify the LSB of selected integer coefficients
          5. Dequantize (multiply back by Q) → float coefficients
          6. IDCT → pixel values
          7. Save using qtables= with the EXACT same Q tables

        When Pillow re-quantizes on save with the same tables, it recovers
        the integer values we set — making the embedding stable.

    Args:
        image_path  : source JPEG (or lossy WebP)
        message     : UTF-8 string to embed
        output_path : destination path (saved as JPEG)

    Returns:
        dict: bits_used, capacity_bits, payload_pct, quality
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

    # Get quantization tables from source — these are our embedding key
    qtables, quality = _get_jpeg_qtables(image_path)
    if qtables is None:
        qtables = _default_qtables(85)
        quality = 85
        print("[DCT EMBED] No Q tables found in source — using standard Q85 tables.")

    luma_qtable = qtables.get(0, list(_default_qtables(quality)[0]))

    # Load image, convert to YCbCr, work on Y (luma) channel
    pil_img     = Image.open(image_path).convert("RGB")
    ycbcr       = pil_img.convert("YCbCr")
    ycbcr_arr   = np.array(ycbcr, dtype=np.uint8)
    y_orig      = ycbcr_arr[:, :, 0].astype(np.float64)
    orig_h, orig_w = y_orig.shape

    # Build payload: header + message bits + terminator
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

    # Split Y channel into 8x8 blocks (centered at 0 for DCT)
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

                Q = _q_value(luma_qtable, row, col)

                # Step 1: Quantize the float DCT coefficient to an integer
                q_coeff = int(round(dct_block[row, col] / Q))

                # Step 2: Modify the LSB of the quantized integer
                q_coeff = (q_coeff & ~1) | payload[bit_index]

                # Step 3: Dequantize back to float domain
                dct_block[row, col] = float(q_coeff * Q)

                bit_index += 1

            # IDCT the modified block back to pixel domain
            blocks[r, c] = _idct_2d(dct_block)

    # Reconstruct Y channel, clip to valid range
    y_modified = _reconstruct_channel(blocks + 128.0, orig_h, orig_w)
    y_modified  = np.clip(np.round(y_modified), 0, 255).astype(np.uint8)

    # Rebuild YCbCr array with modified Y channel
    out_ycbcr = ycbcr_arr.copy()
    out_ycbcr[:, :, 0] = y_modified

    # Convert back to RGB for saving
    result_rgb = Image.fromarray(out_ycbcr, mode="YCbCr").convert("RGB")

    # Enforce JPEG output
    out_path = Path(output_path)
    if out_path.suffix.lower() not in (".jpg", ".jpeg"):
        out_path = out_path.with_suffix(".jpg")
        print(f"[DCT EMBED] Output changed to {out_path} (DCT requires JPEG).")

    # Save with the EXACT same Q tables from the source.
    # This is the critical step — Pillow will re-quantize using these tables
    # and recover the integer values we embedded.
    result_rgb.save(
        str(out_path),
        format   = "JPEG",
        qtables  = qtables,
        subsampling = 0,   # 4:4:4 — no chroma subsampling for fidelity
    )

    payload_pct = (total_bits / capacity_bits) * 100
    print(f"[DCT EMBED] Embedded {total_bits} bits ({payload_pct:.2f}% capacity).")
    print(f"[DCT EMBED] Q tables preserved from source. Est. quality: {quality}.")

    return {
        "bits_used"    : total_bits,
        "capacity_bits": capacity_bits,
        "payload_pct"  : payload_pct,
        "quality"      : quality,
    }


def decode_dct(image_path: str) -> str:
    """
    Extract a hidden message from a DCT-embedded JPEG image.

    Mirrors the embed process exactly:
        1. Read Q tables from the stego image
        2. DCT each 8x8 block of the Y channel
        3. Quantize with the SAME Q tables
        4. Read LSBs of the quantized integers at the same positions
        5. Parse header, find terminator, decode UTF-8

    Args:
        image_path: path to the stego JPEG

    Returns:
        Decoded message string.

    Raises:
        ValueError: structured, human-readable error for every failure mode.
    """
    from core.format_handler import classify, ImageFormat

    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    info = classify(image_path)
    if info.actual_format not in (ImageFormat.JPEG, ImageFormat.WEBP):
        raise ValueError(
            f"DCT decoder requires JPEG or WebP. Got: {info.actual_format.value}."
        )

    # Read Q tables — must match the tables used during embedding
    qtables, _ = _get_jpeg_qtables(image_path)
    if qtables is None:
        raise ValueError(
            "Could not read quantization tables from this image. "
            "The image may not be a standard JPEG, or it has been "
            "processed in a way that removed the Q tables."
        )

    luma_qtable = qtables.get(0, list(_default_qtables(85)[0]))

    # Load Y channel
    pil_img     = Image.open(image_path).convert("RGB")
    ycbcr       = pil_img.convert("YCbCr")
    y_arr       = np.array(ycbcr)[:, :, 0].astype(np.float64)
    orig_h, orig_w = y_arr.shape

    blocks, _, _ = _get_channel_blocks(y_arr - 128.0)
    bh, bw = blocks.shape[0], blocks.shape[1]

    # Extract bits by reading LSBs of quantized coefficients
    extracted_bits = []
    for r in range(bh):
        for c in range(bw):
            dct_block = _dct_2d(blocks[r, c])
            for (row, col) in MID_FREQ_POSITIONS:
                Q         = _q_value(luma_qtable, row, col)
                q_coeff   = int(round(dct_block[row, col] / Q))
                extracted_bits.append(q_coeff & 1)

    if len(extracted_bits) < 16:
        raise ValueError(
            "Image too small to contain a valid DCT-embedded message."
        )

    # Validate header
    try:
        method_id, format_code = _parse_header(extracted_bits[:16])
    except ValueError as e:
        raise ValueError(
            f"Header unreadable or corrupted. "
            f"This image may not contain a hidden message, "
            f"or was recompressed after encoding. Detail: {e}"
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

    # Search for terminator at byte-aligned positions (skip 16-bit header)
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
            f"If recompressed at a different quality setting or format-converted "
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