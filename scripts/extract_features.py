"""
Feature Extraction Script — Scry Phase 8

Runs extract_features() on every image in data/prepared/
and saves the full feature matrix to data/features.npz

Output:
    data/features.npz — compressed archive containing:
        X        : float64 array (n_samples x 24) — feature matrix
        y        : str array    (n_samples,)       — image type labels
        splits   : str array    (n_samples,)       — "train" or "test"
        paths    : str array    (n_samples,)       — relative file paths
        names    : str array    (24,)              — feature names

    data/feature_extraction_log.txt — skipped files and errors

Usage:
    python extract_features.py
    python extract_features.py --prepared data/prepared --output data/features.npz
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time
import traceback

import numpy as np
from PIL import Image

from ml.type_features import extract_features, FeatureVector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PREPARED_DIR = Path("data/prepared")
OUTPUT_PATH  = Path("data/features.npz")
LOG_PATH     = Path("data/feature_extraction_log.txt")

IMAGE_TYPES = [
    "photographic",
    "scanned",
    "ai_generated",
    "screenshot",
    "synthetic",
]

SPLITS = ["train", "test"]


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
# Main extraction
# ---------------------------------------------------------------------------

def extract(prepared_dir: Path, output_path: Path, log_path: Path) -> None:

    # -----------------------------------------------------------------------
    # Collect all image paths from prepared/train/ and prepared/test/
    # -----------------------------------------------------------------------

    all_entries = []   # list of (path, image_type, split)

    for split in SPLITS:
        for image_type in IMAGE_TYPES:
            type_dir = prepared_dir / split / image_type
            if not type_dir.exists():
                continue
            files = sorted(type_dir.glob("*.png"))
            for f in files:
                all_entries.append((f, image_type, split))

    total = len(all_entries)
    if total == 0:
        print("ERROR: No images found in prepared directory.")
        print(f"  Expected structure: {prepared_dir}/{{train,test}}/{{type}}/*.png")
        return

    print(f"  Found {total} images across {len(SPLITS)} splits and {len(IMAGE_TYPES)} types")
    print()

    # -----------------------------------------------------------------------
    # Per-type count summary before starting
    # -----------------------------------------------------------------------

    type_split_counts: dict[str, dict[str, int]] = {
        t: {"train": 0, "test": 0} for t in IMAGE_TYPES
    }
    for _, image_type, split in all_entries:
        type_split_counts[image_type][split] += 1

    print(f"  {'Type':<18} {'Train':>7} {'Test':>7} {'Total':>7}")
    print(f"  {'-'*18} {'-'*7} {'-'*7} {'-'*7}")
    for t in IMAGE_TYPES:
        tr = type_split_counts[t]["train"]
        te = type_split_counts[t]["test"]
        if tr + te > 0:
            print(f"  {t:<18} {tr:>7} {te:>7} {tr+te:>7}")
    print()

    # -----------------------------------------------------------------------
    # Extract features
    # -----------------------------------------------------------------------

    feature_rows  : list[np.ndarray] = []
    label_rows    : list[str]        = []
    split_rows    : list[str]        = []
    path_rows     : list[str]        = []
    skipped       : list[str]        = []

    start_time = time.time()

    for i, (img_path, image_type, split) in enumerate(all_entries):
        try:
            with Image.open(img_path) as img:
                arr = np.array(img.convert("RGB"), dtype=np.uint8)

            fv = extract_features(arr)
            feature_rows.append(fv.to_array())
            label_rows.append(image_type)
            split_rows.append(split)
            path_rows.append(str(img_path.relative_to(prepared_dir)))

        except Exception as e:
            skipped.append(f"{img_path} — {e}\n{traceback.format_exc()}")

        progress_bar(i + 1, total, prefix="  Extracting")

    elapsed = time.time() - start_time
    print()

    # -----------------------------------------------------------------------
    # Build arrays
    # -----------------------------------------------------------------------

    X      = np.array(feature_rows, dtype=np.float64)
    y      = np.array(label_rows,   dtype=object)
    splits = np.array(split_rows,   dtype=object)
    paths  = np.array(path_rows,    dtype=object)
    names  = np.array(FeatureVector.feature_names(), dtype=object)

    ok = len(feature_rows)

    # -----------------------------------------------------------------------
    # Save .npz
    # -----------------------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X      = X,
        y      = y,
        splits = splits,
        paths  = paths,
        names  = names,
    )

    # -----------------------------------------------------------------------
    # Write log
    # -----------------------------------------------------------------------

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Feature Extraction Log\n")
        f.write(f"Processed : {ok}/{total}\n")
        f.write(f"Skipped   : {len(skipped)}\n")
        f.write(f"Elapsed   : {elapsed:.1f}s\n\n")
        if skipped:
            f.write("SKIPPED FILES:\n")
            for entry in skipped:
                f.write(f"\n{entry}\n{'─'*60}\n")
        else:
            f.write("No files skipped.\n")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("=" * 55)
    print("  EXTRACTION SUMMARY")
    print("=" * 55)
    print(f"  Processed        {ok}/{total}")
    print(f"  Skipped          {len(skipped)}")
    print(f"  Feature matrix   {X.shape[0]} × {X.shape[1]}")
    print(f"  Time elapsed     {elapsed:.1f}s")
    print(f"  Output           {output_path.resolve()}")
    print(f"  Log              {log_path.resolve()}")
    print()

    # Per-type in final matrix
    print(f"  {'Type':<18} {'Count':>7} {'% of total':>12}")
    print(f"  {'-'*18} {'-'*7} {'-'*12}")
    for t in IMAGE_TYPES:
        count = int(np.sum(y == t))
        if count > 0:
            pct = 100 * count / len(y)
            print(f"  {t:<18} {count:>7} {pct:>11.1f}%")

    print("=" * 55)

    # Feature sanity check — flag any features with zero variance
    zero_var = [
        FeatureVector.feature_names()[i]
        for i in range(X.shape[1])
        if np.var(X[:, i]) < 1e-10
    ]
    if zero_var:
        print()
        print(f"  ⚠ WARNING: {len(zero_var)} feature(s) have near-zero variance:")
        for name in zero_var:
            print(f"    • {name}")
        print("  These features will not contribute to classification.")
    else:
        print()
        print("  ✓ All 24 features have non-zero variance — feature extraction healthy.")

    print(f"\n✓ Feature extraction complete → {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract features from prepared dataset for Scry Phase 8."
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=PREPARED_DIR,
        help=f"Path to prepared dataset directory (default: {PREPARED_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output .npz file path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=LOG_PATH,
        help=f"Log file path (default: {LOG_PATH})",
    )
    args = parser.parse_args()

    print("Scry — Feature Extraction Script")
    print(f"  Prepared dir : {args.prepared}")
    print(f"  Output       : {args.output}")
    print()

    extract(args.prepared, args.output, args.log)