"""
Model Training Script — Scry Phase 8

Loads data/features.npz and trains five Random Forest classifiers:
    model_general.pkl       — trained on all types combined
    model_photographic.pkl  — specialised for photographic images
    model_scanned.pkl       — specialised for scanned images
    model_ai.pkl            — specialised for AI-generated images
    model_screenshot.pkl    — specialised for screenshot images
    model_synthetic.pkl     — specialised for synthetic images

Each type-specific model is a binary classifier:
    "is this image of type X, or not?"
This makes each model an expert at detecting its own type's
statistical signature rather than trying to do all 5 at once.

The general model is a 5-class classifier used as the primary
inference path. Type-specific models are used to boost confidence
when the general model is uncertain.

Output:
    data/models/model_general.pkl
    data/models/model_photographic.pkl
    data/models/model_scanned.pkl
    data/models/model_ai.pkl
    data/models/model_screenshot.pkl
    data/models/model_synthetic.pkl
    data/models/training_report.txt

Usage:
    python scripts/train_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

from ml.type_features import FeatureVector, ALL_IMAGE_TYPES, ImageType

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURES_PATH = Path("data/features.npz")
MODELS_DIR    = Path("data/models")
REPORT_PATH   = Path("data/models/training_report.txt")

# Plan targets — Phase 8 acceptance criteria
ACCURACY_TARGETS = {
    "photographic" : 0.94,
    "scanned"      : 0.85,
    "ai_generated" : 0.80,
    "screenshot"   : 0.75,
    "synthetic"    : 0.80,
    "general"      : 0.82,
}

RF_PARAMS = dict(
    n_estimators     = 300,
    max_features     = "sqrt",
    min_samples_leaf = 2,
    random_state     = 42,
    n_jobs           = -1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(path: Path):
    print(f"  Loading features from {path}...")
    data   = np.load(path, allow_pickle=True)
    X      = data["X"].astype(np.float64)
    y      = data["y"].astype(str)
    splits = data["splits"].astype(str)
    print(f"  Loaded {X.shape[0]} samples × {X.shape[1]} features")
    print()
    return X, y, splits


def train_general_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, dict]:
    """
    Train a 5-class Random Forest on all image types.
    Returns (model_bundle, metrics).
    """
    print("  Training general model (5-class)...")
    t0 = time.time()

    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train_enc)

    y_pred    = rf.predict(X_test)
    accuracy  = float(accuracy_score(y_test_enc, y_pred))
    elapsed   = time.time() - t0

    # Per-class accuracy
    per_class = {}
    for i, label in enumerate(le.classes_):
        mask = y_test_enc == i
        if mask.any():
            per_class[label] = float(accuracy_score(
                y_test_enc[mask], y_pred[mask]
            ))

    # Confusion matrix
    cm = confusion_matrix(y_test_enc, y_pred, labels=list(range(len(le.classes_))))

    print(f"    ✓ Accuracy: {accuracy:.4f}  ({elapsed:.1f}s)")

    bundle = {
        "model"           : {"rf": rf, "le": le},
        "model_version"   : "1.0",
        "feature_version" : "1.0",
        "model_type"      : "general",
        "training_info"   : {
            "total_samples"     : len(y_train) + len(y_test),
            "n_train"           : len(y_train),
            "n_test"            : len(y_test),
            "accuracy"          : accuracy,
            "per_class_accuracy": per_class,
            "label_counts"      : {
                t: int(np.sum(y_train == t)) for t in ALL_IMAGE_TYPES
            },
        },
    }

    metrics = {
        "accuracy"          : accuracy,
        "per_class_accuracy": per_class,
        "confusion_matrix"  : cm,
        "classes"           : list(le.classes_),
        "elapsed"           : elapsed,
    }

    return bundle, metrics


def train_type_specific_model(
    image_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, dict]:
    """
    Train a binary classifier: is this image of `image_type` or not?
    Returns (model_bundle, metrics).
    """
    print(f"  Training {image_type} model (binary)...")
    t0 = time.time()

    # Convert to binary labels
    y_train_bin = (y_train == image_type).astype(int)
    y_test_bin  = (y_test  == image_type).astype(int)

    # For binary models, oversample the positive class if imbalanced
    pos_count = int(np.sum(y_train_bin == 1))
    neg_count = int(np.sum(y_train_bin == 0))
    ratio     = neg_count / max(pos_count, 1)

    rf = RandomForestClassifier(
        **RF_PARAMS,
        class_weight = "balanced" if ratio > 3 else None,
    )
    rf.fit(X_train, y_train_bin)

    y_pred   = rf.predict(X_test)
    y_proba  = rf.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test_bin, y_pred))
    elapsed  = time.time() - t0

    # True positive rate (recall for the positive class)
    pos_mask = y_test_bin == 1
    tpr = float(accuracy_score(y_test_bin[pos_mask], y_pred[pos_mask])) \
          if pos_mask.any() else 0.0

    # True negative rate
    neg_mask = y_test_bin == 0
    tnr = float(accuracy_score(y_test_bin[neg_mask], y_pred[neg_mask])) \
          if neg_mask.any() else 0.0

    print(f"    ✓ Accuracy: {accuracy:.4f}  TPR: {tpr:.4f}  TNR: {tnr:.4f}  ({elapsed:.1f}s)")

    bundle = {
        "model"           : rf,
        "model_version"   : "1.0",
        "feature_version" : "1.0",
        "model_type"      : f"binary_{image_type}",
        "image_type"      : image_type,
        "training_info"   : {
            "total_samples": len(y_train) + len(y_test),
            "n_train"      : len(y_train),
            "n_test"       : len(y_test),
            "n_positive"   : pos_count,
            "n_negative"   : neg_count,
            "accuracy"     : accuracy,
            "tpr"          : tpr,
            "tnr"          : tnr,
        },
    }

    metrics = {
        "accuracy" : accuracy,
        "tpr"      : tpr,
        "tnr"      : tnr,
        "elapsed"  : elapsed,
    }

    return bundle, metrics


def save_model(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    size_kb = path.stat().st_size / 1024
    print(f"    Saved → {path}  ({size_kb:.1f} KB)")


def print_target_comparison(all_metrics: dict) -> list[str]:
    """Print accuracy vs plan targets. Returns list of report lines."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  ACCURACY vs PLAN TARGETS")
    lines.append("=" * 60)
    lines.append(f"  {'Model':<20} {'Achieved':>10} {'Target':>10} {'Status':>10}")
    lines.append(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    all_passed = True
    for key, metrics in all_metrics.items():
        achieved = metrics.get("accuracy", 0.0)
        target   = ACCURACY_TARGETS.get(key, 0.80)
        passed   = achieved >= target
        status   = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        line = f"  {key:<20} {achieved:>9.1%} {target:>9.1%} {status:>10}"
        lines.append(line)

    lines.append("=" * 60)
    if all_passed:
        lines.append("  ✓ All models meet Phase 8 accuracy targets.")
    else:
        lines.append("  ✗ Some models below target — consider more training data.")
    lines.append("=" * 60)

    for line in lines:
        print(line)

    return lines


def write_report(
    all_metrics   : dict,
    general_metrics: dict,
    report_path   : Path,
    total_elapsed : float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Scry Phase 8 — Model Training Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total training time : {total_elapsed:.1f}s\n\n")

        # General model confusion matrix
        f.write("GENERAL MODEL — CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        classes = general_metrics["classes"]
        cm      = general_metrics["confusion_matrix"]
        header  = f"{'':>15}" + "".join(f"{c[:8]:>10}" for c in classes)
        f.write(header + "\n")
        for i, row_label in enumerate(classes):
            row = f"{row_label[:15]:>15}" + "".join(f"{cm[i,j]:>10}" for j in range(len(classes)))
            f.write(row + "\n")
        f.write("\n")

        # General model per-class accuracy
        f.write("GENERAL MODEL — PER-CLASS ACCURACY\n")
        f.write("-" * 40 + "\n")
        for label, acc in general_metrics["per_class_accuracy"].items():
            target = ACCURACY_TARGETS.get(label, 0.80)
            status = "PASS" if acc >= target else "FAIL"
            f.write(f"  {label:<20} {acc:.4f}  (target {target:.2f})  {status}\n")
        f.write("\n")

        # Type-specific models
        f.write("TYPE-SPECIFIC MODELS\n")
        f.write("-" * 40 + "\n")
        for key, metrics in all_metrics.items():
            if key == "general":
                continue
            f.write(f"  {key:<20} acc={metrics['accuracy']:.4f}  "
                    f"tpr={metrics['tpr']:.4f}  tnr={metrics['tnr']:.4f}\n")
        f.write("\n")

        # Feature importances from general model
        f.write("FEATURE IMPORTANCES (general model)\n")
        f.write("-" * 40 + "\n")
        f.write("  (see model bundle for full importances)\n")

    print(f"\n  Report saved → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(features_path: Path, models_dir: Path) -> None:
    print("Scry — Model Training Script")
    print()

    # Load features
    X, y, splits = load_features(features_path)

    # Split into train / test
    train_mask = splits == "train"
    test_mask  = splits == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(f"  Train set : {len(X_train)} samples")
    print(f"  Test set  : {len(X_test)} samples")
    print()

    all_metrics   = {}
    total_start   = time.time()

    # -----------------------------------------------------------------------
    # General model
    # -----------------------------------------------------------------------

    print("── General Model ─────────────────────────────────────")
    bundle, metrics = train_general_model(X_train, y_train, X_test, y_test)
    save_model(bundle, models_dir / "model_general.pkl")
    all_metrics["general"] = metrics
    general_metrics = metrics
    print()

    # -----------------------------------------------------------------------
    # Type-specific binary models
    # -----------------------------------------------------------------------

    print("── Type-Specific Models ──────────────────────────────")
    type_model_names = {
        "photographic" : "model_photographic.pkl",
        "scanned"      : "model_scanned.pkl",
        "ai_generated" : "model_ai.pkl",
        "screenshot"   : "model_screenshot.pkl",
        "synthetic"    : "model_synthetic.pkl",
    }

    for image_type, filename in type_model_names.items():
        bundle, metrics = train_type_specific_model(
            image_type, X_train, y_train, X_test, y_test
        )
        save_model(bundle, models_dir / filename)
        all_metrics[image_type] = metrics
        print()

    total_elapsed = time.time() - total_start

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------

    print_target_comparison(all_metrics)

    write_report(all_metrics, general_metrics, REPORT_PATH, total_elapsed)

    print(f"\n  Total time : {total_elapsed:.1f}s")
    print(f"  Models dir : {models_dir.resolve()}")
    print("\n✓ Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(FEATURES_PATH, MODELS_DIR)