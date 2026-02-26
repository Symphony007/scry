"""
Stego Dataset Generator — Scry Phase 8 (ML Stego Detection)

Takes every clean image in data/prepared/ and creates stego versions
using LSB Replacement, LSB Matching, and DWT at 10%, 25%, and 50% payload.

Metadata method is intentionally excluded — it makes zero pixel changes
and is statistically undetectable by any pixel-based ML model.
This limitation is documented in KNOWN_ISSUES.md.

Output structure:
    data/stego_prepared/
    ├── train/
    │   ├── clean/          ← copied from data/prepared/train/
    │   ├── lsb_replace/
    │   ├── lsb_match/
    │   └── dwt/
    ├── test/
    │   └── ...
    └── stego_labels.csv   ← filepath, image_type, split, method, payload_pct, label

Label encoding:
    0 = clean
    1 = stego

Usage:
    python scripts/generate_stego_dataset.py
    python scripts/generate_stego_dataset.py --workers 4
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import random
import shutil
import string
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

from core.embedder              import embed as lsb_replace
from core.lsb_matching_embedder import embed_matching
from core.dwt_embedder          import embed_dwt, get_dwt_capacity
from core.utils                 import calculate_capacity

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PREPARED_DIR  = Path("data/prepared")
OUTPUT_DIR    = Path("data/stego_prepared")
LABELS_PATH   = Path("data/stego_prepared/stego_labels.csv")

IMAGE_TYPES   = ["photographic", "scanned", "ai_generated", "screenshot", "synthetic"]
SPLITS        = ["train", "test"]

METHODS       = ["lsb_replace", "lsb_match", "dwt"]
PAYLOADS      = [0.10, 0.25, 0.50]   # fraction of capacity

SEED          = 42
MIN_IMAGE_DIM = 32   # skip images smaller than this
MIN_DWT_BITS  = 24   # skip DWT jobs where capacity is too small to be useful


# ---------------------------------------------------------------------------
# Message generation
# ---------------------------------------------------------------------------

def _make_message(capacity_bytes: int, payload_fraction: float) -> str:
    """
    Generate a random ASCII message that fills exactly payload_fraction
    of the given capacity. Returns at least 1 character.
    """
    n_bytes = max(1, int(capacity_bytes * payload_fraction))
    chars   = string.ascii_letters + string.digits + " .,!?"
    rng     = random.Random()
    return "".join(rng.choice(chars) for _ in range(n_bytes))


# ---------------------------------------------------------------------------
# Per-image embed worker
# ---------------------------------------------------------------------------

def _embed_one(args: tuple) -> dict | None:
    """
    Embed a single image with a single method at a single payload level.
    Returns a label dict on success, None on failure.
    Designed to run in a subprocess via ProcessPoolExecutor.
    """
    (
        src_path,
        dst_path,
        method,
        payload_fraction,
        image_type,
        split,
    ) = args

    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        arr = np.array(Image.open(src_path).convert("RGB"), dtype=np.uint8)
        h, w = arr.shape[:2]

        if h < MIN_IMAGE_DIM or w < MIN_IMAGE_DIM:
            return None

        # --- Method-specific capacity calculation ---
        if method == "dwt":
            cap_bits  = get_dwt_capacity(arr)
            if cap_bits < MIN_DWT_BITS:
                # Image too small to embed anything useful via DWT
                return None
            cap_bytes = cap_bits // 8
        else:
            capacity  = calculate_capacity(arr)
            cap_bytes = capacity["usable_bytes"]

        message = _make_message(max(1, cap_bytes), payload_fraction)

        # Save a temp PNG for methods that need a file path
        tmp_src = str(dst_path).replace(".png", "_tmpsrc.png")
        Image.fromarray(arr).save(tmp_src, format="PNG")

        try:
            if method == "lsb_replace":
                lsb_replace(tmp_src, message, str(dst_path))

            elif method == "lsb_match":
                embed_matching(tmp_src, message, str(dst_path))

            elif method == "dwt":
                result   = embed_dwt(tmp_src, message, str(dst_path))
                dst_path = Path(result["output_path"])

        finally:
            Path(tmp_src).unlink(missing_ok=True)

        return {
            "filepath"    : str(dst_path.relative_to(OUTPUT_DIR)),
            "image_type"  : image_type,
            "split"       : split,
            "method"      : method,
            "payload_pct" : payload_fraction * 100,
            "label"       : 1,
        }

    except Exception:
        return None


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
# Main
# ---------------------------------------------------------------------------

def generate(
    prepared_dir : Path,
    output_dir   : Path,
    labels_path  : Path,
    n_workers    : int = 1,
) -> None:

    print("Scry — Stego Dataset Generator")
    print(f"  Source    : {prepared_dir}")
    print(f"  Output    : {output_dir}")
    print(f"  Methods   : {METHODS}")
    print(f"  Payloads  : {[f'{int(p*100)}%' for p in PAYLOADS]}")
    print(f"  Workers   : {n_workers}")
    print()

    # -----------------------------------------------------------------------
    # Collect all clean source images
    # -----------------------------------------------------------------------

    clean_entries = []
    for split in SPLITS:
        for image_type in IMAGE_TYPES:
            type_dir = prepared_dir / split / image_type
            if not type_dir.exists():
                continue
            for f in sorted(type_dir.glob("*.png")):
                clean_entries.append((f, image_type, split))

    if not clean_entries:
        print("ERROR: No clean images found. Run prepare_dataset.py first.")
        return

    print(f"  Found {len(clean_entries)} clean images")
    print()

    # -----------------------------------------------------------------------
    # Build clean label rows first
    # -----------------------------------------------------------------------

    all_labels = []

    for src, image_type, split in clean_entries:
        dst = output_dir / split / "clean" / image_type / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not dst.exists():
            shutil.copy2(src, dst)

        all_labels.append({
            "filepath"    : str(dst.relative_to(output_dir)),
            "image_type"  : image_type,
            "split"       : split,
            "method"      : "clean",
            "payload_pct" : 0.0,
            "label"       : 0,
        })

    print(f"  ✓ Clean images copied  ({len(all_labels)} entries)")

    # -----------------------------------------------------------------------
    # Build embed job list
    # -----------------------------------------------------------------------

    jobs = []
    for src, image_type, split in clean_entries:
        for method in METHODS:
            for payload in PAYLOADS:
                stem     = src.stem
                dst_name = f"{stem}_{method}_p{int(payload*100):03d}.png"
                dst_path = output_dir / split / method / image_type / dst_name
                jobs.append((
                    src, dst_path, method, payload, image_type, split
                ))

    total_jobs = len(jobs)
    print(f"  Embedding {total_jobs} stego images "
          f"({len(METHODS)} methods × {len(PAYLOADS)} payloads × {len(clean_entries)} clean)")
    print()

    # -----------------------------------------------------------------------
    # Run embed jobs
    # -----------------------------------------------------------------------

    start_time = time.time()
    done       = 0
    skipped    = 0

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_embed_one, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_labels.append(result)
                else:
                    skipped += 1
                done += 1
                progress_bar(done, total_jobs, prefix="  Embedding")
    else:
        for job in jobs:
            result = _embed_one(job)
            if result:
                all_labels.append(result)
            else:
                skipped += 1
            done += 1
            progress_bar(done, total_jobs, prefix="  Embedding")

    elapsed = time.time() - start_time
    print()

    # -----------------------------------------------------------------------
    # Write labels CSV
    # -----------------------------------------------------------------------

    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filepath", "image_type", "split", "method", "payload_pct", "label"]
        )
        writer.writeheader()
        writer.writerows(all_labels)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    stego_count = sum(1 for r in all_labels if r["label"] == 1)
    clean_count = sum(1 for r in all_labels if r["label"] == 0)

    print("=" * 60)
    print("  STEGO DATASET SUMMARY")
    print("=" * 60)
    print(f"  Clean images   : {clean_count}")
    print(f"  Stego images   : {stego_count}")
    print(f"  Skipped        : {skipped}")
    print(f"  Total labels   : {len(all_labels)}")
    print(f"  Time elapsed   : {elapsed:.1f}s")
    print(f"  Labels CSV     : {labels_path.resolve()}")
    print()

    print(f"  {'Method':<15} {'Count':>8}")
    print(f"  {'-'*15} {'-'*8}")
    for method in ["clean"] + METHODS:
        count = sum(1 for r in all_labels if r["method"] == method)
        print(f"  {method:<15} {count:>8}")

    print("=" * 60)
    print("\n✓ Stego dataset generation complete.")
    print("  Next step: python scripts/extract_stego_features.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate stego training dataset for Scry ML stego detection."
    )
    parser.add_argument("--prepared", type=Path, default=PREPARED_DIR)
    parser.add_argument("--output",   type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers",  type=int,  default=1,
                        help="Parallel workers. Use 4-8 on multicore machines.")
    args = parser.parse_args()

    generate(
        prepared_dir = args.prepared,
        output_dir   = OUTPUT_DIR,
        labels_path  = LABELS_PATH,
        n_workers    = args.workers,
    )