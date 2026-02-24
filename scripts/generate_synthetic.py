"""
Synthetic Image Generator — Scry Phase 8 Training Data

Generates 750 synthetic images across 10 categories and saves them
into data/benchmark_images/by_type/synthetic/

Categories:
    solid_color         — uniform fills
    linear_gradient     — horizontal/vertical/diagonal gradients
    radial_gradient     — circular gradients from center
    noise_random        — pure random pixel noise
    noise_gaussian      — gaussian noise on a base color
    checkerboard        — alternating black/white blocks
    geometric_shapes    — rectangles and circles on solid backgrounds
    stripes             — horizontal/vertical/diagonal stripe patterns
    test_card           — color bar test patterns
    concentric          — concentric rings/squares

Usage:
    python generate_synthetic.py
    python generate_synthetic.py --output data/benchmark_images/by_type/synthetic --count 750
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = Path("data/benchmark_images/by_type/synthetic")
DEFAULT_COUNT  = 750
IMAGE_SIZE     = (256, 256)   # all synthetic images are 256x256 PNG

CATEGORIES = [
    "solid_color",
    "linear_gradient",
    "radial_gradient",
    "noise_random",
    "noise_gaussian",
    "checkerboard",
    "geometric_shapes",
    "stripes",
    "test_card",
    "concentric",
]


# ---------------------------------------------------------------------------
# Generators — each returns an RGB numpy array (H x W x 3, uint8)
# ---------------------------------------------------------------------------

def gen_solid_color(rng: np.random.Generator) -> np.ndarray:
    """Uniform solid color fill."""
    color = rng.integers(0, 256, 3, dtype=np.uint8)
    return np.full((*IMAGE_SIZE, 3), color, dtype=np.uint8)


def gen_linear_gradient(rng: np.random.Generator) -> np.ndarray:
    """Linear gradient between two random colors."""
    h, w      = IMAGE_SIZE
    c1        = rng.integers(0, 256, 3).astype(np.float64)
    c2        = rng.integers(0, 256, 3).astype(np.float64)
    direction = rng.integers(0, 3)  # 0=horizontal, 1=vertical, 2=diagonal

    arr = np.zeros((h, w, 3), dtype=np.float64)

    if direction == 0:
        t = np.linspace(0, 1, w)
        for ch in range(3):
            arr[:, :, ch] = c1[ch] + (c2[ch] - c1[ch]) * t
    elif direction == 1:
        t = np.linspace(0, 1, h)
        for ch in range(3):
            arr[:, :, ch] = (c1[ch] + (c2[ch] - c1[ch]) * t)[:, np.newaxis]
    else:
        tx = np.linspace(0, 1, w)
        ty = np.linspace(0, 1, h)
        t  = (tx[np.newaxis, :] + ty[:, np.newaxis]) / 2
        for ch in range(3):
            arr[:, :, ch] = c1[ch] + (c2[ch] - c1[ch]) * t

    return np.clip(arr, 0, 255).astype(np.uint8)


def gen_radial_gradient(rng: np.random.Generator) -> np.ndarray:
    """Radial gradient from center outward."""
    h, w = IMAGE_SIZE
    c1   = rng.integers(0, 256, 3).astype(np.float64)
    c2   = rng.integers(0, 256, 3).astype(np.float64)
    cx   = rng.integers(w // 4, 3 * w // 4)
    cy   = rng.integers(h // 4, 3 * h // 4)

    xs   = np.arange(w) - cx
    ys   = np.arange(h) - cy
    xx, yy = np.meshgrid(xs, ys)
    dist   = np.sqrt(xx ** 2 + yy ** 2)
    max_d  = np.sqrt(cx ** 2 + cy ** 2) + 1
    t      = np.clip(dist / max_d, 0, 1)

    arr = np.zeros((h, w, 3), dtype=np.float64)
    for ch in range(3):
        arr[:, :, ch] = c1[ch] + (c2[ch] - c1[ch]) * t

    return np.clip(arr, 0, 255).astype(np.uint8)


def gen_noise_random(rng: np.random.Generator) -> np.ndarray:
    """Pure random pixel noise — maximum entropy."""
    return rng.integers(0, 256, (*IMAGE_SIZE, 3), dtype=np.uint8)


def gen_noise_gaussian(rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise on a random base color."""
    h, w  = IMAGE_SIZE
    base  = rng.integers(30, 220, 3).astype(np.float64)
    sigma = float(rng.integers(15, 60))
    noise = rng.normal(0, sigma, (h, w, 3))
    arr   = np.full((h, w, 3), base, dtype=np.float64) + noise
    return np.clip(arr, 0, 255).astype(np.uint8)


def gen_checkerboard(rng: np.random.Generator) -> np.ndarray:
    """Checkerboard pattern with random cell size and colors."""
    h, w      = IMAGE_SIZE
    cell_size = int(rng.integers(8, 48))
    c1        = rng.integers(0, 256, 3, dtype=np.uint8)
    c2        = rng.integers(0, 256, 3, dtype=np.uint8)

    rows = np.arange(h) // cell_size
    cols = np.arange(w) // cell_size
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    mask   = ((rr + cc) % 2 == 0)

    arr = np.where(mask[:, :, np.newaxis], c1, c2).astype(np.uint8)
    return arr


def gen_geometric_shapes(rng: np.random.Generator) -> np.ndarray:
    """Random rectangles and ellipses on a solid background."""
    h, w = IMAGE_SIZE
    bg   = tuple(rng.integers(200, 256, 3).tolist())
    img  = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)

    n_shapes = int(rng.integers(3, 12))
    for _ in range(n_shapes):
        color   = tuple(rng.integers(0, 200, 3).tolist())
        x1      = int(rng.integers(0, w - 10))
        y1      = int(rng.integers(0, h - 10))
        x2      = int(rng.integers(x1 + 10, min(x1 + 120, w)))
        y2      = int(rng.integers(y1 + 10, min(y1 + 120, h)))
        filled  = bool(rng.integers(0, 2))
        fill    = color if filled else None
        outline = color

        if rng.integers(0, 2) == 0:
            draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=2)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=fill, outline=outline, width=2)

    return np.array(img, dtype=np.uint8)


def gen_stripes(rng: np.random.Generator) -> np.ndarray:
    """Stripe pattern — horizontal, vertical, or diagonal."""
    h, w      = IMAGE_SIZE
    stripe_w  = int(rng.integers(4, 32))
    direction = rng.integers(0, 3)
    n_colors  = int(rng.integers(2, 5))
    colors    = rng.integers(0, 256, (n_colors, 3), dtype=np.uint8)

    arr = np.zeros((h, w, 3), dtype=np.uint8)

    if direction == 0:   # horizontal
        indices      = (np.arange(h) // stripe_w) % n_colors
        arr[:, :]    = colors[indices][:, np.newaxis, :]
    elif direction == 1: # vertical
        indices      = (np.arange(w) // stripe_w) % n_colors
        arr[:, :]    = colors[indices][np.newaxis, :, :]
    else:                # diagonal
        r_idx = np.arange(h)[:, np.newaxis]
        c_idx = np.arange(w)[np.newaxis, :]
        idx   = ((r_idx + c_idx) // stripe_w) % n_colors
        arr   = colors[idx]

    return arr


def gen_test_card(rng: np.random.Generator) -> np.ndarray:
    """Color bar test card pattern."""
    h, w = IMAGE_SIZE
    arr  = np.zeros((h, w, 3), dtype=np.uint8)

    color_bars = np.array([
        [255, 255, 255],
        [255, 255,   0],
        [  0, 255, 255],
        [  0, 255,   0],
        [255,   0, 255],
        [255,   0,   0],
        [  0,   0, 255],
        [  0,   0,   0],
    ], dtype=np.uint8)

    n_bars   = int(rng.integers(4, 9))
    indices  = rng.choice(len(color_bars), n_bars, replace=False)
    selected = color_bars[indices]
    bar_w    = w // n_bars

    for i, color in enumerate(selected):
        start       = i * bar_w
        end         = start + bar_w if i < n_bars - 1 else w
        arr[:, start:end] = color

    # Grayscale ramp at the bottom
    gray_h = h // 6
    ramp   = np.linspace(0, 255, w).astype(np.uint8)
    arr[h - gray_h:, :, 0] = ramp
    arr[h - gray_h:, :, 1] = ramp
    arr[h - gray_h:, :, 2] = ramp

    return arr


def gen_concentric(rng: np.random.Generator) -> np.ndarray:
    """Concentric rings or squares."""
    h, w       = IMAGE_SIZE
    cx, cy     = w // 2, h // 2
    ring_width = int(rng.integers(8, 32))
    shape      = rng.integers(0, 2)   # 0=circles, 1=squares
    n_colors   = int(rng.integers(2, 5))
    colors     = rng.integers(0, 256, (n_colors, 3), dtype=np.uint8)

    r_idx = np.arange(h)[:, np.newaxis]
    c_idx = np.arange(w)[np.newaxis, :]

    if shape == 0:
        dist = np.sqrt((r_idx - cy) ** 2 + (c_idx - cx) ** 2).astype(int)
    else:
        dist = np.maximum(np.abs(r_idx - cy), np.abs(c_idx - cx))

    idx = (dist // ring_width) % n_colors
    return colors[idx].astype(np.uint8)


# ---------------------------------------------------------------------------
# Generator dispatch table
# ---------------------------------------------------------------------------

GENERATORS = {
    "solid_color"     : gen_solid_color,
    "linear_gradient" : gen_linear_gradient,
    "radial_gradient" : gen_radial_gradient,
    "noise_random"    : gen_noise_random,
    "noise_gaussian"  : gen_noise_gaussian,
    "checkerboard"    : gen_checkerboard,
    "geometric_shapes": gen_geometric_shapes,
    "stripes"         : gen_stripes,
    "test_card"       : gen_test_card,
    "concentric"      : gen_concentric,
}


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate(output_dir: Path, total_count: int, seed: int = 42) -> None:
    """
    Generate `total_count` synthetic images distributed evenly across
    all categories and save them as PNG files.

    Args:
        output_dir  : destination folder (created if not exists)
        total_count : total number of images to generate
        seed        : RNG seed for reproducibility
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    per_category = total_count // len(CATEGORIES)
    remainder    = total_count  % len(CATEGORIES)

    counts = {cat: per_category for cat in CATEGORIES}
    for cat in CATEGORIES[:remainder]:
        counts[cat] += 1

    total_generated = 0

    for category, count in counts.items():
        gen_fn = GENERATORS[category]
        print(f"  Generating {count:>4} × {category}...")
        for i in range(count):
            arr      = gen_fn(rng)
            filename = f"{category}_{i:04d}.png"
            Image.fromarray(arr).save(output_dir / filename, format="PNG")
            total_generated += 1

    print(f"\n✓ Generated {total_generated} synthetic images → {output_dir}")
    print("\nBreakdown:")
    for cat, count in counts.items():
        print(f"  {cat:<20} {count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic training images for Scry Phase 8."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Total number of images to generate (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    print(f"Scry — Synthetic Image Generator")
    print(f"Output : {args.output}")
    print(f"Count  : {args.count}")
    print(f"Seed   : {args.seed}")
    print()

    generate(args.output, args.count, args.seed)