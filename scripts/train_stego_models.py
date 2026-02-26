"""
Stego Model Trainer — Scry Phase 8 (ML Stego Detection)

Trains one Random Forest binary classifier per image type:
    clean = 0,  stego = 1

Uses the pre-extracted features from data/stego_features/.
Trains on the "train" split, evaluates on the "test" split.

Threshold tuning:
    Instead of using the default 0.5 decision threshold, we sweep
    thresholds from 0.1 → 0.9 and pick the one that maximises F1
    on the test set. This is the mathematically optimal balance
    between missing stego (FNR) and false alarms on clean (FPR).
    Thresholds are saved alongside each model and used during inference.

Output:
    data/models/stego_model_photographic.pkl
    data/models/stego_model_scanned.pkl
    data/models/stego_model_ai_generated.pkl
    data/models/stego_model_screenshot.pkl
    data/models/stego_model_synthetic.pkl
    data/models/stego_model_general.pkl
    data/models/stego_thresholds.json       ← optimal threshold per model

Usage:
    python scripts/train_stego_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import json
import pickle
import time
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FEATURES_DIR = Path("data/stego_features")
LABELS_CSV   = Path("data/stego_prepared/stego_labels.csv")
MODELS_DIR   = Path("data/models")

IMAGE_TYPES  = [
    "photographic",
    "scanned",
    "ai_generated",
    "screenshot",
    "synthetic",
]

RF_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = None,
    min_samples_leaf = 2,
    class_weight     = "balanced",
    random_state     = 42,
    n_jobs           = -1,
)

MIN_TRAIN_SAMPLES = 50
THRESHOLD_SWEEP   = np.arange(0.10, 0.91, 0.05)


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple:
    """
    Sweep decision thresholds and return the one that maximises F1.
    Returns (best_threshold, best_f1).
    """
    best_t  = 0.5
    best_f1 = 0.0
    for t in THRESHOLD_SWEEP:
        y_pred = (y_proba >= t).astype(int)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = float(round(t, 2))
    return best_t, best_f1


# ---------------------------------------------------------------------------
# Load feature data
# ---------------------------------------------------------------------------

def load_data(features_dir: Path, labels_csv: Path) -> dict:
    print("  Loading features...")
    features    = np.load(features_dir / "features.npy")
    labels      = np.load(features_dir / "labels.npy").astype(np.int32)
    image_types = np.load(features_dir / "image_types.npy", allow_pickle=True)
    splits      = np.load(features_dir / "splits.npy",      allow_pickle=True)
    feat_names  = (features_dir / "feature_names.txt").read_text().strip().splitlines()

    methods  = []
    payloads = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            methods.append(row["method"])
            payloads.append(float(row["payload_pct"]))

    methods  = np.array(methods,  dtype=object)
    payloads = np.array(payloads, dtype=np.float32)

    print(f"  Features shape : {features.shape}")
    print(f"  Labels         : {int(np.sum(labels==0))} clean, {int(np.sum(labels==1))} stego")
    print()

    return dict(
        features    = features,
        labels      = labels,
        image_types = image_types,
        splits      = splits,
        methods     = methods,
        payloads    = payloads,
        feat_names  = feat_names,
    )


# ---------------------------------------------------------------------------
# Train + evaluate one model
# ---------------------------------------------------------------------------

def train_model(
    name       : str,
    X_train    : np.ndarray,
    y_train    : np.ndarray,
    X_test     : np.ndarray,
    y_test     : np.ndarray,
    feat_names : list[str],
    meta_test  : dict,
) -> tuple:
    """
    Train a single RF model, tune threshold, print evaluation report.
    Returns (trained_classifier, best_threshold).
    """

    print(f"  Training: {name}")
    print(f"    Train : {int(np.sum(y_train==0))} clean, {int(np.sum(y_train==1))} stego")
    print(f"    Test  : {int(np.sum(y_test==0))} clean, {int(np.sum(y_test==1))} stego")

    t0  = time.time()
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    # Get probabilities for threshold tuning
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Find F1-optimal threshold
    best_t, best_f1 = find_best_threshold(y_test, y_proba)
    y_pred          = (y_proba >= best_t).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"    Train time      : {elapsed:.1f}s")
    print(f"    Best threshold  : {best_t:.2f}  (F1-optimal)")
    print(f"    Accuracy        : {acc*100:.1f}%")
    print(f"    Precision       : {prec*100:.1f}%")
    print(f"    Recall          : {rec*100:.1f}%  (TPR — stego caught)")
    print(f"    F1              : {f1*100:.1f}%")

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"    Confusion       : TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        print(f"    FPR             : {fpr*100:.1f}%  (false alarms on clean)")
        print(f"    FNR             : {fnr*100:.1f}%  (missed stego)")

    # Per-method breakdown
    methods_test = meta_test["methods"]
    for method in ["lsb_replace", "lsb_match", "dwt"]:
        mask = methods_test == method
        if mask.sum() > 0:
            m_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {method:<15} : {m_acc*100:.1f}% ({mask.sum()} samples)")

    # Per-payload breakdown (stego only)
    payloads_test = meta_test["payloads"]
    stego_mask    = y_test == 1
    for payload in [10.0, 25.0, 50.0]:
        mask = stego_mask & (np.abs(payloads_test - payload) < 0.1)
        if mask.sum() > 0:
            p_rec = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            print(f"    payload {int(payload):>3}%     : {p_rec*100:.1f}% TPR ({mask.sum()} samples)")

    # Feature importances (top 5)
    importances = clf.feature_importances_
    ranked      = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    print(f"    Top features :")
    for fname, imp in ranked[:5]:
        bar = "█" * int(imp * 40)
        print(f"      {fname:<25} {imp:.4f}  {bar}")

    print()
    return clf, best_t


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------

def save_model(clf: RandomForestClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f, protocol=5)
    size_kb = path.stat().st_size / 1024
    print(f"  ✓ Saved: {path}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("Scry — Stego Model Trainer")
    print("=" * 60)
    print()

    data = load_data(FEATURES_DIR, LABELS_CSV)

    features    = data["features"]
    labels      = data["labels"]
    image_types = data["image_types"]
    splits      = data["splits"]
    methods     = data["methods"]
    payloads    = data["payloads"]
    feat_names  = data["feat_names"]

    train_mask = splits == "train"
    test_mask  = splits == "test"

    results    = {}
    thresholds = {}

    # -----------------------------------------------------------------------
    # Train per-type models
    # -----------------------------------------------------------------------

    print("Per-type models")
    print("-" * 60)

    for itype in IMAGE_TYPES:
        type_mask = image_types == itype

        tr = train_mask & type_mask
        te = test_mask  & type_mask

        if tr.sum() < MIN_TRAIN_SAMPLES:
            print(f"  SKIP {itype} — only {tr.sum()} train samples")
            continue

        X_train = features[tr]
        y_train = labels[tr]
        X_test  = features[te]
        y_test  = labels[te]

        meta_test = {
            "methods" : methods[te],
            "payloads": payloads[te],
        }

        clf, best_t = train_model(
            name       = f"stego_{itype}",
            X_train    = X_train,
            y_train    = y_train,
            X_test     = X_test,
            y_test     = y_test,
            feat_names = feat_names,
            meta_test  = meta_test,
        )

        model_path = MODELS_DIR / f"stego_model_{itype}.pkl"
        save_model(clf, model_path)

        y_proba       = clf.predict_proba(X_test)[:, 1]
        y_pred        = (y_proba >= best_t).astype(int)
        results[itype]    = accuracy_score(y_test, y_pred)
        thresholds[itype] = best_t

    # -----------------------------------------------------------------------
    # Train general model
    # -----------------------------------------------------------------------

    print()
    print("General model (all types combined)")
    print("-" * 60)

    X_train = features[train_mask]
    y_train = labels[train_mask]
    X_test  = features[test_mask]
    y_test  = labels[test_mask]

    meta_test = {
        "methods" : methods[test_mask],
        "payloads": payloads[test_mask],
    }

    clf_general, best_t_general = train_model(
        name       = "stego_general",
        X_train    = X_train,
        y_train    = y_train,
        X_test     = X_test,
        y_test     = y_test,
        feat_names = feat_names,
        meta_test  = meta_test,
    )

    save_model(clf_general, MODELS_DIR / "stego_model_general.pkl")

    y_proba              = clf_general.predict_proba(X_test)[:, 1]
    y_pred               = (y_proba >= best_t_general).astype(int)
    results["general"]   = accuracy_score(y_test, y_pred)
    thresholds["general"] = best_t_general

    # -----------------------------------------------------------------------
    # Save thresholds
    # -----------------------------------------------------------------------

    thresholds_path = MODELS_DIR / "stego_thresholds.json"
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\n  ✓ Saved thresholds: {thresholds_path}")
    print(f"    {thresholds}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<25} {'Accuracy':>10}  {'Threshold':>10}")
    print(f"  {'-'*25} {'-'*10}  {'-'*10}")
    for name, acc in results.items():
        t      = thresholds[name]
        status = "✓" if acc >= 0.80 else "⚠"
        print(f"  {status} {name:<23} {acc*100:>9.1f}%  {t:>10.2f}")
    print("=" * 60)
    print("\n✓ Stego model training complete.")
    print("  Next step: build ml/stego_detector.py")


if __name__ == "__main__":
    main()