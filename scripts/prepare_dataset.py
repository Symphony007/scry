"""
Dataset Preparation Script — Scry Phase 8

For each image type in data/benchmark_images/by_type/:
    1. Collect all images (jpg, png, bmp, tiff, webp)
    2. Strip EXIF metadata (privacy)
    3. Convert to RGB PNG at 256x256
    4. Sample down to max_per_type
    5. 80/20 stratified train/test split
    6. Write labels.csv

Output structure:
    data/prepared/
    ├── train/
    │   ├── photographic/
    │   ├── scanned/
    │   ├── ai_generated/
    │   ├── screenshot/
    │   └── synthetic/
    ├── test/
    │   └── ...
    └── labels.csv

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --max 1000 --size 256
"""

import argparse
import csv
import random
import shutil
import time
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOURCE_DIR   = Path("data/benchmark_images/by_type")
OUTPUT_DIR   = Path("data/prepared")
MAX_PER_TYPE = 1000
IMAGE_SIZE   = 256
TRAIN_RATIO  = 0.80
SEED         = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

IMAGE_TYPES = [
    "photographic",
    "scanned",
    "ai_generated",
    "screenshot",
    "synthetic",
]


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

def progress_bar(current: int, total: int, prefix: str = "", width: int = 35) -> None:
    filled = int(width * current / total) if total > 0 else 0
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / total if total > 0 else 0
    print(f"\r  {prefix} [{bar}] {current}/{total} ({pct:.0f}%)", end="", flush=True)
    if current == total:
        print()


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def process_image(src_path: Path, dst_path: Path, size: int) -> bool:
    """
    Load image, strip EXIF, convert to RGB, resize, save as PNG.
    Returns True on success, False on error.
    """
    try:
        with Image.open(src_path) as img:
            rgb = img.convert("RGB")
            rgb = rgb.resize((size, size), Image.LANCZOS)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(dst_path, format="PNG", optimize=False)
        return True
    except Exception as e:
        print(f"\n  [SKIP] {src_path.name} — {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(
    source_dir   : Path,
    output_dir   : Path,
    max_per_type : int,
    image_size   : int,
    train_ratio  : float,
    seed         : int,
) -> None:

    rng = random.Random(seed)

    print(f"  Source   : {source_dir}")
    print(f"  Output   : {output_dir}")
    print(f"  Max/type : {max_per_type}")
    print(f"  Size     : {image_size}x{image_size} px")
    print(f"  Split    : {int(train_ratio*100)}% train / {int((1-train_ratio)*100)}% test")
    print(f"  Seed     : {seed}")
    print()

    if output_dir.exists():
        print(f"  Clearing existing output dir: {output_dir}")
        shutil.rmtree(output_dir)

    labels      = []
    type_counts = {}
    total_start = time.time()

    for image_type in IMAGE_TYPES:
        type_dir = source_dir / image_type

        if not type_dir.exists():
            print(f"  [SKIP] {image_type}/ — folder not found")
            continue

        all_files = sorted([
            p for p in type_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not all_files:
            print(f"  [SKIP] {image_type}/ — no images found")
            continue

        found = len(all_files)

        if found > max_per_type:
            rng.shuffle(all_files)
            selected = all_files[:max_per_type]
        else:
            selected = list(all_files)

        n = len(selected)
        rng.shuffle(selected)
        n_train     = int(n * train_ratio)
        train_files = selected[:n_train]
        test_files  = selected[n_train:]

        print(f"  {image_type:<15} {found:>5} found → {n} selected "
              f"({len(train_files)} train / {len(test_files)} test)")

        ok = 0
        t0 = time.time()

        for i, src in enumerate(train_files):
            dst = output_dir / "train" / image_type / f"{image_type}_{i:05d}.png"
            if process_image(src, dst, image_size):
                labels.append({
                    "filepath"  : str(dst.relative_to(output_dir)),
                    "image_type": image_type,
                    "split"     : "train",
                    "source"    : src.name,
                })
                ok += 1
            progress_bar(i + 1, len(train_files), prefix="    train")

        for i, src in enumerate(test_files):
            dst = output_dir / "test" / image_type / f"{image_type}_{i:05d}.png"
            if process_image(src, dst, image_size):
                labels.append({
                    "filepath"  : str(dst.relative_to(output_dir)),
                    "image_type": image_type,
                    "split"     : "test",
                    "source"    : src.name,
                })
                ok += 1
            progress_bar(i + 1, len(test_files), prefix="    test ")

        elapsed = time.time() - t0
        print(f"    ✓ {ok}/{n} processed in {elapsed:.1f}s")
        type_counts[image_type] = ok
        print()

    labels_path = output_dir / "labels.csv"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "image_type", "split", "source"]
        )
        writer.writeheader()
        writer.writerows(labels)

    total_elapsed = time.time() - total_start

    print("=" * 55)
    print("  DATASET SUMMARY")
    print("=" * 55)
    total = sum(type_counts.values())
    for t, c in type_counts.items():
        print(f"  {t:<18} {c:>5} images")
    print(f"  {'TOTAL':<18} {total:>5} images")
    print(f"  Train / Test     {int(train_ratio*100)}% / {int((1-train_ratio)*100)}%")
    print(f"  labels.csv       {len(labels)} rows")
    print(f"  Time elapsed     {total_elapsed:.1f}s")
    print(f"  Output           {output_dir.resolve()}")
    print("=" * 55)
    print("\n✓ Dataset preparation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare training dataset for Scry Phase 8."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max",    type=int,  default=MAX_PER_TYPE,
                        help="Max images per type (default: 1000)")
    parser.add_argument("--size",   type=int,  default=IMAGE_SIZE,
                        help="Output image size px (default: 256)")
    parser.add_argument("--seed",   type=int,  default=SEED)
    args = parser.parse_args()

    print("Scry — Dataset Preparation Script")
    print()

    prepare(
        source_dir   = args.source,
        output_dir   = args.output,
        max_per_type = args.max,
        image_size   = args.size,
        train_ratio  = TRAIN_RATIO,
        seed         = args.seed,
    )