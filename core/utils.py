from pathlib import Path
import numpy as np
from PIL import Image

# Formats supported for lossless spatial-domain embedding
LOSSLESS_FORMATS = {".png", ".tiff", ".tif"}
# Formats supported for DCT-domain embedding
LOSSY_FORMATS = {".webp", ".jpg", ".jpeg"}
# All supported formats
SUPPORTED_FORMATS = LOSSLESS_FORMATS | LOSSY_FORMATS


def load_image(path: str) -> tuple[np.ndarray, str]:
    """
    Load an image from disk and return it as a NumPy array alongside its
    detected format extension (lowercase, with dot).

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the format is not supported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{suffix}'. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    img = Image.open(path).convert("RGB")
    return np.array(img), suffix


def save_image(array: np.ndarray, path: str) -> None:
    """
    Save a NumPy array as an image to disk.
    Enforces PNG output if the destination extension is a lossy format,
    because LSB embedding cannot survive lossy compression.

    Warns but does not raise when a lossy extension is requested.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in LOSSY_FORMATS:
        new_path = p.with_suffix(".png")
        print(
            f"[WARNING] Requested output format '{suffix}' is lossy and will "
            f"destroy embedded data. Saving as PNG instead: {new_path}"
        )
        path = str(new_path)

    img = Image.fromarray(array.astype(np.uint8))
    img.save(path)


def text_to_bits(text: str) -> list[int]:
    """
    Convert a UTF-8 string to a flat list of bits (ints, 0 or 1).
    Encodes the full string to bytes first, then converts each byte to 8 bits.
    This correctly handles all Unicode characters including multi-byte sequences.
    """
    raw_bytes = text.encode("utf-8")
    bits = []
    for byte in raw_bytes:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def bits_to_text(bits: list[int]) -> str:
    """
    Convert a flat list of bits back to a UTF-8 string.
    Collects ALL bytes first, then decodes the complete byte array as UTF-8.
    Never decodes byte-by-byte — that breaks multi-byte Unicode characters.

    Raises:
        ValueError: if the bit list length is not a multiple of 8.
        UnicodeDecodeError: if the bytes are not valid UTF-8.
    """
    if len(bits) % 8 != 0:
        raise ValueError(
            f"Bit list length {len(bits)} is not a multiple of 8."
        )

    byte_array = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        byte_array.append(byte)

    return byte_array.decode("utf-8")


def calculate_capacity(array: np.ndarray) -> dict:
    """
    Calculate the embedding capacity of an image array.

    Returns a dict with:
        total_bits     : total LSB bits available (pixels * 3 channels)
        total_bytes    : total_bits // 8
        usable_bytes   : total_bytes minus 2 bytes reserved for the terminator
        ascii_chars    : usable bytes (1 byte per ASCII char)
        note           : reminder that Unicode chars may use 2–4 bytes each
    """
    pixels = array.shape[0] * array.shape[1]
    total_bits = pixels * 3  # R, G, B channels
    total_bytes = total_bits // 8
    usable_bytes = total_bytes - 2  # reserve 2 bytes for 16-bit terminator

    return {
        "total_bits"  : total_bits,
        "total_bytes" : total_bytes,
        "usable_bytes": usable_bytes,
        "ascii_chars" : usable_bytes,
        "note"        : "Unicode characters may require 2–4 bytes each.",
    }


def calculate_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    Higher is better. LSB embedding typically yields 50–55 dB.
    Returns float('inf') if the images are identical.

    Args:
        original: the unmodified image array
        modified: the image array after embedding

    Returns:
        PSNR value in dB.
    """
    original = original.astype(np.float64)
    modified = modified.astype(np.float64)

    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return float("inf")

    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))