"""
Stego Feature Extractor — Scry Phase 8 (ML Stego Detection)

Extracts 14 stego-detection features from every image in data/stego_prepared/
and saves them as a numpy array for model training.

Features (14 total):
    --- Original spatial features (kept) ---
    0  rs_asymmetry          — RS Analysis score (R+M vs R-M groups)
    1  chi_square_stat       — Chi-square statistic on LSB pixel pairs
    2  histogram_combing     — Even/odd value pair balance
    3  lsb_entropy           — Shannon entropy of the LSB plane
    4  lsb_pair_balance      — Fraction of adjacent LSB pairs that are equal
    5  block_entropy_var     — Variance of per-block entropy (8x8 blocks)
    6  noise_residual_entropy — Entropy of high-frequency noise residual
    7  pixel_pair_corr       — Pearson correlation of adjacent pixel pairs

    --- New stronger features ---
    8  spa_asymmetry         — Sample Pair Analysis: close-value pair asymmetry
                               Best known detector for LSB matching specifically
    9  close_color_pair_ratio — Ratio of adjacent pixels differing by exactly 1
                               Embedding artificially inflates this value
    10 markov_h_disruption   — Horizontal Markov transition probability disruption
                               Embedding breaks natural horizontal pixel structure
    11 markov_v_disruption   — Vertical Markov transition probability disruption
    12 spam_mean             — SPAM: mean of difference array (subtractive adjacency)
    13 pvp_score             — Pixel Value Pairing: even/odd clustering score
                               Direct signature of LSB replacement

Output:
    data/stego_features/
    ├── features.npy       ← float32 array  (N, 14)
    ├── labels.npy         ← int8 array     (N,)
    ├── image_types.npy    ← str array      (N,)
    └── feature_names.txt

Usage:
    python scripts/extract_stego_features.py
    python scripts/extract_stego_features.py --workers 6
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import csv
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEGO_DIR   = Path("data/stego_prepared")
LABELS_CSV  = Path("data/stego_prepared/stego_labels.csv")
OUTPUT_DIR  = Path("data/stego_features")

FEATURE_NAMES = [
    "rs_asymmetry",
    "chi_square_stat",
    "histogram_combing",
    "lsb_entropy",
    "lsb_pair_balance",
    "block_entropy_var",
    "noise_residual_entropy",
    "pixel_pair_corr",
    "spa_asymmetry",
    "close_color_pair_ratio",
    "markov_h_disruption",
    "markov_v_disruption",
    "spam_mean",
    "pvp_score",
]

N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Original features (kept)
# ---------------------------------------------------------------------------

def _lsb_plane(arr: np.ndarray) -> np.ndarray:
    return arr & 1


def _rs_asymmetry(gray: np.ndarray) -> float:
    def _rs_count(block, mask, flip):
        flipped = block.copy().astype(np.int32)
        if flip:
            for i, m in enumerate(mask.flatten()):
                if m == 1:
                    idx = np.unravel_index(i, block.shape)
                    v = flipped[idx]
                    flipped[idx] = v ^ 1
                elif m == -1:
                    idx = np.unravel_index(i, block.shape)
                    v = flipped[idx]
                    flipped[idx] = v - 1 if v % 2 == 1 else v + 1
        noise = int(np.sum(np.abs(np.diff(flipped.flatten()))))
        orig  = int(np.sum(np.abs(np.diff(block.flatten()))))
        return int(noise > orig), int(noise < orig)

    h, w = gray.shape
    mask = np.array([[1, -1], [1, -1]])
    r_pos = s_pos = r_neg = s_neg = 0
    count = 0
    for row in range(0, h - 1, 2):
        for col in range(0, w - 1, 2):
            block = gray[row:row+2, col:col+2].astype(np.int32)
            if block.shape != (2, 2):
                continue
            rp, sp = _rs_count(block, mask, False)
            rn, sn = _rs_count(block, mask, True)
            r_pos += rp; s_pos += sp
            r_neg += rn; s_neg += sn
            count += 1
    if count == 0:
        return 0.0
    return float(abs((r_pos - r_neg) / count) - abs((s_pos - s_neg) / count))


def _chi_square_stat(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    pairs = []
    for i in range(0, 254, 2):
        expected = (hist[i] + hist[i+1]) / 2.0
        if expected > 0:
            chi = ((hist[i] - expected) ** 2 + (hist[i+1] - expected) ** 2) / expected
            pairs.append(chi)
    return float(np.mean(pairs)) if pairs else 0.0


def _histogram_combing(gray: np.ndarray) -> float:
    flat = gray.flatten()
    even = np.sum(flat % 2 == 0)
    total = len(flat)
    return float(even / total) if total > 0 else 0.5


def _lsb_entropy(gray: np.ndarray) -> float:
    lsb = _lsb_plane(gray).flatten()
    p1  = float(np.mean(lsb))
    p0  = 1.0 - p1
    if p0 <= 0 or p1 <= 0:
        return 0.0
    return float(-p0 * np.log2(p0) - p1 * np.log2(p1))


def _lsb_pair_balance(gray: np.ndarray) -> float:
    lsb   = _lsb_plane(gray)
    left  = lsb[:, :-1].flatten()
    right = lsb[:, 1:].flatten()
    if len(left) == 0:
        return 0.5
    return float(np.mean(left == right))


def _block_entropy_var(gray: np.ndarray, block_size: int = 8) -> float:
    h, w      = gray.shape
    entropies = []
    for row in range(0, h - block_size + 1, block_size):
        for col in range(0, w - block_size + 1, block_size):
            block = gray[row:row+block_size, col:col+block_size].flatten()
            vals, counts = np.unique(block, return_counts=True)
            probs = counts / counts.sum()
            ent   = float(-np.sum(probs * np.log2(probs + 1e-10)))
            entropies.append(ent)
    return float(np.var(entropies)) if len(entropies) > 1 else 0.0


def _noise_residual_entropy(gray: np.ndarray) -> float:
    from scipy.ndimage import median_filter
    residual = np.abs(
        gray.astype(np.float32) - median_filter(gray.astype(np.float32), size=3)
    )
    residual = np.clip(residual, 0, 255).astype(np.uint8)
    vals, counts = np.unique(residual.flatten(), return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def _pixel_pair_corr(gray: np.ndarray) -> float:
    left  = gray[:, :-1].flatten().astype(np.float32)
    right = gray[:, 1:].flatten().astype(np.float32)
    if len(left) < 2:
        return 0.0
    try:
        r, _ = stats.pearsonr(left, right)
        return float(r) if np.isfinite(r) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# New stronger features
# ---------------------------------------------------------------------------

def _spa_asymmetry(gray: np.ndarray) -> float:
    """
    Sample Pair Analysis (SPA) — asymmetry between close-value pixel pairs.

    For each horizontally adjacent pair (u, v):
      - W set: pairs where both values have the same LSB
      - B set: pairs where values differ by 1 (u and u+1 or v and v+1)

    In clean images W ≈ B. LSB matching inflates W relative to B.
    Returns |W - B| / (W + B) — higher = more likely stego.

    Reference: Dumitrescu et al., 2003.
    """
    flat  = gray.astype(np.int32)
    left  = flat[:, :-1]
    right = flat[:, 1:]
    diff  = np.abs(left - right)

    # W: same LSB pairs (diff is even)
    W = int(np.sum(diff % 2 == 0))
    # B: adjacent value pairs (diff == 1)
    B = int(np.sum(diff == 1))

    total = W + B
    if total == 0:
        return 0.0
    return float(abs(W - B) / total)


def _close_color_pair_ratio(gray: np.ndarray) -> float:
    """
    Ratio of horizontally adjacent pixel pairs that differ by exactly 1.

    Embedding (especially LSB replacement) creates artificial ±1 differences
    between neighbouring pixels that wouldn't normally exist in natural images.
    Clean images: low ratio. Stego images: elevated ratio.
    """
    flat  = gray.astype(np.int32)
    left  = flat[:, :-1]
    right = flat[:, 1:]
    diff  = np.abs(left - right)
    total = diff.size
    if total == 0:
        return 0.0
    return float(np.sum(diff == 1) / total)


def _markov_disruption(gray: np.ndarray, direction: str = "h") -> float:
    """
    Markov transition probability disruption.

    Natural images follow approximately Markovian pixel statistics.
    Embedding disrupts transition probabilities between value pairs.

    We use a coarse 8-bin quantisation for efficiency.
    Measures: sum of squared deviations from uniform transitions.
    Higher = more disrupted = more likely stego.

    direction: "h" = horizontal, "v" = vertical
    """
    arr = gray.astype(np.int32) // 32   # 0-7 bins (256 / 32 = 8)

    if direction == "h":
        left  = arr[:, :-1].flatten()
        right = arr[:, 1:].flatten()
    else:
        left  = arr[:-1, :].flatten()
        right = arr[1:, :].flatten()

    n_bins = 8
    # Build transition matrix
    T = np.zeros((n_bins, n_bins), dtype=np.float32)
    np.add.at(T, (left, right), 1)

    # Normalise rows
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    T = T / row_sums

    # Disruption = deviation from uniform (1/n_bins per cell)
    uniform   = 1.0 / n_bins
    disruption = float(np.sum((T - uniform) ** 2))
    return disruption


def _spam_mean(gray: np.ndarray) -> float:
    """
    SPAM (Subtractive Pixel Adjacency Model) — mean of horizontal difference array.

    Computes D = |x[i] - x[i+1]| for all adjacent pairs, then returns mean(D).
    Embedding adds noise that shifts this distribution.
    Clean images have higher mean (natural gradients).
    Heavy stego has lower mean (embedding smooths differences).

    Reference: Pevny, Bas, Fridrich, 2010.
    """
    flat = gray.astype(np.int32)
    diff = np.abs(flat[:, :-1] - flat[:, 1:]).flatten()
    return float(np.mean(diff)) if len(diff) > 0 else 0.0


def _pvp_score(gray: np.ndarray) -> float:
    """
    Pixel Value Pairing (PVP) score — direct LSB replacement signature.

    LSB replacement forces each pixel into a (value, value^1) pair.
    Measures: how much the histogram clusters into even/odd pairs
    compared to a natural distribution.

    Score = mean of |hist[2k] - hist[2k+1]| / mean(hist)
    Clean images: ~0.5 (natural variation between pairs).
    LSB replacement: → 0.0 (pairs forced to equal counts).
    Returns 1 - normalised_score so higher = more stego.
    """
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    mean_h  = float(np.mean(hist)) + 1e-10
    pair_diffs = []
    for i in range(0, 256, 2):
        pair_diffs.append(abs(int(hist[i]) - int(hist[i+1])) / mean_h)
    score = float(np.mean(pair_diffs))
    # Invert: lower pair_diff = more stego
    return float(1.0 / (1.0 + score))


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    Extract all 14 stego detection features from an RGB image array.

    Args:
        arr : (H, W, 3) uint8

    Returns:
        float32 array of shape (14,)
    """
    gray = np.array(Image.fromarray(arr).convert("L"), dtype=np.uint8)

    features = np.array([
        _rs_asymmetry(gray),              # 0
        _chi_square_stat(gray),            # 1
        _histogram_combing(gray),          # 2
        _lsb_entropy(gray),                # 3
        _lsb_pair_balance(gray),           # 4
        _block_entropy_var(gray),          # 5
        _noise_residual_entropy(gray),     # 6
        _pixel_pair_corr(gray),            # 7
        _spa_asymmetry(gray),              # 8  NEW
        _close_color_pair_ratio(gray),     # 9  NEW
        _markov_disruption(gray, "h"),     # 10 NEW
        _markov_disruption(gray, "v"),     # 11 NEW
        _spam_mean(gray),                  # 12 NEW
        _pvp_score(gray),                  # 13 NEW
    ], dtype=np.float32)

    features = np.where(np.isfinite(features), features, 0.0)
    return features


# ---------------------------------------------------------------------------
# Worker (subprocess)
# ---------------------------------------------------------------------------

def _process_row(args: tuple) -> tuple | None:
    filepath, label, image_type, split, stego_dir = args
    try:
        arr   = np.array(Image.open(stego_dir / filepath).convert("RGB"), dtype=np.uint8)
        feats = extract_features(arr)
        return (feats, int(label), image_type, split)
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

def main(stego_dir: Path, labels_csv: Path, output_dir: Path, n_workers: int) -> None:

    print("Scry — Stego Feature Extractor")
    print(f"  Source CSV : {labels_csv}")
    print(f"  Output     : {output_dir}")
    print(f"  Features   : {N_FEATURES}")
    print(f"  Workers    : {n_workers}")
    print()

    if not labels_csv.exists():
        print("ERROR: stego_labels.csv not found. Run generate_stego_dataset.py first.")
        return

    rows = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                row["filepath"],
                int(row["label"]),
                row["image_type"],
                row["split"],
                stego_dir,
            ))

    total = len(rows)
    print(f"  Loaded {total} rows from CSV")
    print()

    all_features = []
    all_labels   = []
    all_types    = []
    all_splits   = []
    skipped      = 0
    start_time   = time.time()

    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_process_row, row): row for row in rows}
            done    = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    feats, label, itype, split = result
                    all_features.append(feats)
                    all_labels.append(label)
                    all_types.append(itype)
                    all_splits.append(split)
                else:
                    skipped += 1
                done += 1
                progress_bar(done, total, prefix="  Extracting")
    else:
        for i, row in enumerate(rows):
            result = _process_row(row)
            if result:
                feats, label, itype, split = result
                all_features.append(feats)
                all_labels.append(label)
                all_types.append(itype)
                all_splits.append(split)
            else:
                skipped += 1
            progress_bar(i + 1, total, prefix="  Extracting")

    elapsed = time.time() - start_time
    print()

    if not all_features:
        print("ERROR: No features extracted.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    features_arr = np.array(all_features, dtype=np.float32)
    labels_arr   = np.array(all_labels,   dtype=np.int8)
    types_arr    = np.array(all_types,    dtype=object)
    splits_arr   = np.array(all_splits,   dtype=object)

    np.save(output_dir / "features.npy",    features_arr)
    np.save(output_dir / "labels.npy",      labels_arr)
    np.save(output_dir / "image_types.npy", types_arr)
    np.save(output_dir / "splits.npy",      splits_arr)

    (output_dir / "feature_names.txt").write_text(
        "\n".join(FEATURE_NAMES), encoding="utf-8"
    )

    clean_count = int(np.sum(labels_arr == 0))
    stego_count = int(np.sum(labels_arr == 1))

    print("=" * 60)
    print("  FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Total rows       : {total}")
    print(f"  Extracted        : {len(all_features)}")
    print(f"  Skipped          : {skipped}")
    print(f"  Clean samples    : {clean_count}")
    print(f"  Stego samples    : {stego_count}")
    print(f"  Feature shape    : {features_arr.shape}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    print()
    print(f"  {'Image Type':<20} {'Count':>8}")
    print(f"  {'-'*20} {'-'*8}")
    for itype in sorted(set(all_types)):
        count = all_types.count(itype)
        print(f"  {itype:<20} {count:>8}")
    print("=" * 60)
    print("\n✓ Feature extraction complete.")
    print("  Next step: python scripts/train_stego_models.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stego-dir",  type=Path, default=STEGO_DIR)
    parser.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    parser.add_argument("--output",     type=Path, default=OUTPUT_DIR)
    parser.add_argument("--workers",    type=int,  default=1)
    args = parser.parse_args()

    main(
        stego_dir  = args.stego_dir,
        labels_csv = args.labels_csv,
        output_dir = args.output,
        n_workers  = args.workers,
    )