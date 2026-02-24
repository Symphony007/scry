"""
Model Calibration Script — Scry Phase 8

Applies Platt scaling (logistic regression) to each trained model's
raw probability outputs so that a 70% score actually means the model
is correct ~70% of the time.

Why this matters:
    Random Forests are known to produce overconfident probabilities —
    they push scores toward 0.0 and 1.0 more than is warranted.
    Without calibration, a score of 0.85 might only correspond to
    ~65% actual accuracy. This is the root cause of the "20-30%
    confidence" problem in the detection pipeline.

Method:
    Platt scaling fits a logistic regression on top of the model's
    raw predict_proba() outputs using a held-out calibration set.
    This is lightweight, fast, and well-understood.

Output:
    data/models/model_general_calibrated.pkl
    data/models/model_photographic_calibrated.pkl
    data/models/model_scanned_calibrated.pkl
    data/models/model_ai_calibrated.pkl
    data/models/model_screenshot_calibrated.pkl
    data/models/model_synthetic_calibrated.pkl
    data/models/calibration_report.txt

Usage:
    python scripts/calibrate_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import time
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import LabelEncoder

from ml.type_features import FeatureVector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURES_PATH = Path("data/features.npz")
MODELS_DIR    = Path("data/models")
REPORT_PATH   = Path("data/models/calibration_report.txt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(path: Path):
    data   = np.load(path, allow_pickle=True)
    X      = data["X"].astype(np.float64)
    y      = data["y"].astype(str)
    splits = data["splits"].astype(str)
    return X, y, splits


def load_model(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(bundle: dict, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    size_kb = path.stat().st_size / 1024
    print(f"    Saved → {path.name}  ({size_kb:.1f} KB)")


def brier_score_multiclass(y_true_enc, proba, n_classes):
    """Mean Brier score across all classes for multiclass problems."""
    scores = []
    for c in range(n_classes):
        y_bin = (y_true_enc == c).astype(float)
        scores.append(brier_score_loss(y_bin, proba[:, c]))
    return float(np.mean(scores))


def reliability_summary(y_true_bin, proba_pos, n_bins=10):
    """
    Compute mean calibration error — average |predicted prob - actual freq|
    across probability bins. Lower is better. <0.05 is well calibrated.
    """
    try:
        fraction_pos, mean_pred = calibration_curve(
            y_true_bin, proba_pos, n_bins=n_bins, strategy="uniform"
        )
        return float(np.mean(np.abs(fraction_pos - mean_pred)))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# General model calibration (multiclass)
# ---------------------------------------------------------------------------

def calibrate_general(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bundle: dict,
) -> dict:
    print("  Calibrating general model (multiclass Platt scaling)...")
    t0 = time.time()

    rf = bundle["model"]["rf"]
    le = bundle["model"]["le"]

    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)

    # Raw probabilities before calibration
    proba_raw_test = rf.predict_proba(X_test)
    brier_before   = brier_score_multiclass(y_test_enc, proba_raw_test, len(le.classes_))

    # Fit one logistic regression per class (one-vs-rest Platt scaling)
    # Use training set predictions to fit calibrators
    proba_train = rf.predict_proba(X_train)

    calibrators = []
    for c in range(len(le.classes_)):
        y_bin = (y_train_enc == c).astype(float)
        lr    = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(proba_train, y_bin)
        calibrators.append(lr)

    # Apply calibration to test set and renormalize
    proba_cal_test = np.zeros_like(proba_raw_test)
    for c, lr in enumerate(calibrators):
        proba_cal_test[:, c] = lr.predict_proba(proba_raw_test)[:, 1]

    # Renormalize rows to sum to 1
    row_sums = proba_cal_test.sum(axis=1, keepdims=True)
    proba_cal_test = proba_cal_test / np.maximum(row_sums, 1e-8)

    brier_after = brier_score_multiclass(y_test_enc, proba_cal_test, len(le.classes_))

    elapsed = time.time() - t0
    improvement = brier_before - brier_after

    print(f"    Brier score  before: {brier_before:.4f}")
    print(f"    Brier score  after : {brier_after:.4f}  "
          f"({'↓ improved' if improvement > 0 else '↑ worse'} by {abs(improvement):.4f})")
    print(f"    ✓ Done ({elapsed:.1f}s)")

    # Store calibrators in the bundle
    bundle["calibrators"]     = calibrators
    bundle["calibrated"]      = True
    bundle["calibration_info"] = {
        "method"       : "platt_ovr",
        "brier_before" : brier_before,
        "brier_after"  : brier_after,
        "improvement"  : improvement,
    }

    return bundle, {
        "brier_before": brier_before,
        "brier_after" : brier_after,
        "improvement" : improvement,
    }


# ---------------------------------------------------------------------------
# Binary model calibration
# ---------------------------------------------------------------------------

def calibrate_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bundle: dict,
) -> tuple[dict, dict]:
    image_type = bundle["image_type"]
    print(f"  Calibrating {image_type} model (binary Platt scaling)...")
    t0 = time.time()

    rf          = bundle["model"]
    y_train_bin = (y_train == image_type).astype(int)
    y_test_bin  = (y_test  == image_type).astype(int)

    # Raw probabilities before calibration
    proba_raw_train = rf.predict_proba(X_train)[:, 1]
    proba_raw_test  = rf.predict_proba(X_test)[:, 1]

    brier_before = brier_score_loss(y_test_bin, proba_raw_test)
    mce_before   = reliability_summary(y_test_bin, proba_raw_test)

    # Fit logistic regression on training set raw probabilities
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(proba_raw_train.reshape(-1, 1), y_train_bin)

    # Apply to test set
    proba_cal_test = lr.predict_proba(proba_raw_test.reshape(-1, 1))[:, 1]

    brier_after = brier_score_loss(y_test_bin, proba_cal_test)
    mce_after   = reliability_summary(y_test_bin, proba_cal_test)

    elapsed     = time.time() - t0
    improvement = brier_before - brier_after

    print(f"    Brier score  before: {brier_before:.4f}  MCE: {mce_before:.4f}")
    print(f"    Brier score  after : {brier_after:.4f}  MCE: {mce_after:.4f}  "
          f"({'↓ improved' if improvement > 0 else '↑ worse'} by {abs(improvement):.4f})")
    print(f"    ✓ Done ({elapsed:.1f}s)")

    bundle["calibrator"]      = lr
    bundle["calibrated"]      = True
    bundle["calibration_info"] = {
        "method"       : "platt_binary",
        "brier_before" : brier_before,
        "brier_after"  : brier_after,
        "mce_before"   : mce_before,
        "mce_after"    : mce_after,
        "improvement"  : improvement,
    }

    return bundle, {
        "brier_before": brier_before,
        "brier_after" : brier_after,
        "mce_before"  : mce_before,
        "mce_after"   : mce_after,
        "improvement" : improvement,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(all_metrics: dict, report_path: Path, elapsed: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Scry Phase 8 — Calibration Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total time : {elapsed:.1f}s\n\n")
        f.write(f"{'Model':<22} {'Brier Before':>14} {'Brier After':>13} "
                f"{'Improvement':>13}\n")
        f.write("-" * 65 + "\n")
        for name, m in all_metrics.items():
            f.write(f"  {name:<20} {m['brier_before']:>13.4f} "
                    f"{m['brier_after']:>13.4f} "
                    f"{m['improvement']:>+13.4f}\n")
        f.write("\n")
        f.write("Brier score: lower is better. 0.0 = perfect, 0.25 = random.\n")
        f.write("MCE (mean calibration error): <0.05 = well calibrated.\n")
    print(f"\n  Report saved → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def calibrate(features_path: Path, models_dir: Path) -> None:
    print("Scry — Model Calibration Script")
    print()

    X, y, splits = load_features(features_path)

    train_mask = splits == "train"
    test_mask  = splits == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"  Train : {len(X_train)} samples")
    print(f"  Test  : {len(X_test)} samples")
    print()

    all_metrics = {}
    total_start = time.time()

    # -----------------------------------------------------------------------
    # General model
    # -----------------------------------------------------------------------

    print("── General Model ─────────────────────────────────────")
    bundle = load_model(models_dir / "model_general.pkl")
    bundle, metrics = calibrate_general(X_train, y_train, X_test, y_test, bundle)
    save_model(bundle, models_dir / "model_general_calibrated.pkl")
    all_metrics["general"] = metrics
    print()

    # -----------------------------------------------------------------------
    # Binary models
    # -----------------------------------------------------------------------

    print("── Type-Specific Models ──────────────────────────────")
    binary_models = [
        ("model_photographic.pkl", "model_photographic_calibrated.pkl"),
        ("model_scanned.pkl",      "model_scanned_calibrated.pkl"),
        ("model_ai.pkl",           "model_ai_calibrated.pkl"),
        ("model_screenshot.pkl",   "model_screenshot_calibrated.pkl"),
        ("model_synthetic.pkl",    "model_synthetic_calibrated.pkl"),
    ]

    for src_name, dst_name in binary_models:
        bundle = load_model(models_dir / src_name)
        bundle, metrics = calibrate_binary(X_train, y_train, X_test, y_test, bundle)
        save_model(bundle, models_dir / dst_name)
        all_metrics[bundle["image_type"]] = metrics
        print()

    total_elapsed = time.time() - total_start

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("=" * 65)
    print("  CALIBRATION SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<22} {'Brier Before':>14} {'Brier After':>13} {'Δ':>8}")
    print(f"  {'-'*22} {'-'*14} {'-'*13} {'-'*8}")
    for name, m in all_metrics.items():
        delta = m["improvement"]
        arrow = "↓" if delta > 0 else "↑"
        print(f"  {name:<22} {m['brier_before']:>13.4f} "
              f"{m['brier_after']:>13.4f} "
              f"{arrow}{abs(delta):>6.4f}")
    print("=" * 65)
    print("  Brier score: lower = better. 0.0 = perfect, 0.25 = random guess.")
    print("=" * 65)

    write_report(all_metrics, REPORT_PATH, total_elapsed)

    print(f"\n  Total time : {total_elapsed:.1f}s")
    print("\n✓ Calibration complete.")
    print("  Next step: wire calibrated models into ml/classifier.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    calibrate(FEATURES_PATH, MODELS_DIR)