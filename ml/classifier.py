"""
Hierarchical Inference Classifier — Scry Phase 8

Wires the general model and 5 type-specific binary models into a single
classify() call used by the rest of the pipeline.

Inference logic:
    1. Run general model → 5-class probability distribution
    2. If top class confidence >= HIGH_CONFIDENCE_THRESHOLD (0.70):
       → Trust the general model, return result directly
    3. If confidence < threshold:
       → Run all type-specific binary models
       → Blend general + binary scores (60% general, 40% binary)
       → Return blended result with method="ensemble"
    4. If no model is loaded:
       → Fall back to rule-based classification from type_classifier.py

This mirrors the architecture described in the Phase 8 plan.

Usage:
    from ml.classifier import HierarchicalClassifier

    clf = HierarchicalClassifier()
    clf.load()   # loads from data/models/ automatically

    result = clf.classify(image_array)
    print(result.image_type)       # "photographic", "scanned" etc.
    print(result.confidence)       # 0.0 - 1.0
    print(result.weight_table)     # detector weights for this type
"""

import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from ml.type_features import (
    extract_features, FeatureVector,
    ImageType, ALL_IMAGE_TYPES,
)
from ml.type_classifier import (
    ClassificationResult, WEIGHT_TABLES, RELIABILITY_NOTES,
    _classify_by_rules,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODELS_DIR         = Path("data/models")
HIGH_CONFIDENCE_THRESHOLD  = 0.70   # above this → trust general model alone
BLEND_GENERAL_WEIGHT       = 0.60   # weight for general model in ensemble
BLEND_BINARY_WEIGHT        = 0.40   # weight for binary models in ensemble

# Map from lowercase data label → ImageType constant (for weight table lookup)
LABEL_TO_IMAGE_TYPE = {
    "photographic" : ImageType.PHOTOGRAPHIC,
    "scanned"      : ImageType.SCANNED,
    "ai_generated" : ImageType.AI_GENERATED,
    "screenshot"   : ImageType.SCREENSHOT,
    "synthetic"    : ImageType.SYNTHETIC,
    "unknown"      : ImageType.UNKNOWN,
}

MODEL_FILES = {
    "general"      : "model_general.pkl",
    "photographic" : "model_photographic.pkl",
    "scanned"      : "model_scanned.pkl",
    "ai_generated" : "model_ai.pkl",
    "screenshot"   : "model_screenshot.pkl",
    "synthetic"    : "model_synthetic.pkl",
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class HierarchicalClassifier:
    """
    Two-stage hierarchical image type classifier.

    Stage 1 — General model (5-class Random Forest):
        Fast classification into one of 5 types.
        Used when confidence >= HIGH_CONFIDENCE_THRESHOLD.

    Stage 2 — Binary ensemble (5 one-vs-rest Random Forests):
        Each model is an expert at detecting one image type.
        Activated when general model is uncertain.
        Scores are blended with general model output.

    Fallback — Rule-based classifier:
        Used when no trained models are available.
        Confidence capped at 0.75, labelled "rules".
    """

    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR):
        self.models_dir     = Path(models_dir)
        self._general       = None   # {"rf": ..., "le": ...}
        self._binary        = {}     # image_type -> sklearn RF
        self._loaded_types  = []
        self._is_loaded     = False

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def load(self, models_dir: Path = None) -> bool:
        """
        Load all models from models_dir.
        Returns True if at least the general model loaded successfully.
        Partial loads are accepted — missing binary models are silently skipped.
        """
        if models_dir:
            self.models_dir = Path(models_dir)

        # Load general model
        general_path = self.models_dir / MODEL_FILES["general"]
        if not general_path.exists():
            print(f"[CLASSIFIER] General model not found at {general_path}. "
                  f"Falling back to rule-based classification.")
            return False

        try:
            bundle          = _load_pkl(general_path)
            self._general   = bundle["model"]   # {"rf": ..., "le": ...}
            self._is_loaded = True
            print(f"[CLASSIFIER] General model loaded.")
        except Exception as e:
            print(f"[CLASSIFIER] Failed to load general model: {e}")
            return False

        # Load binary models — non-fatal if missing
        for image_type, filename in MODEL_FILES.items():
            if image_type == "general":
                continue
            path = self.models_dir / filename
            if not path.exists():
                continue
            try:
                bundle = _load_pkl(path)
                self._binary[image_type] = bundle["model"]   # sklearn RF
                self._loaded_types.append(image_type)
            except Exception as e:
                print(f"[CLASSIFIER] Could not load {filename}: {e}")

        print(f"[CLASSIFIER] Loaded general model + "
              f"{len(self._loaded_types)} binary models "
              f"({', '.join(self._loaded_types) or 'none'}).")
        return True

    # -----------------------------------------------------------------------
    # Classify
    # -----------------------------------------------------------------------

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        Classify an image into one of the five image types.

        Args:
            image: RGB numpy array (H x W x 3, uint8)

        Returns:
            ClassificationResult with type, confidence, weights, and notes.
        """
        fv = extract_features(image)

        if not self._is_loaded:
            # Rule-based fallback
            image_type, confidence, class_probs = _classify_by_rules(fv)
            method = "rules"
        else:
            image_type, confidence, class_probs, method = self._hierarchical_inference(fv)

        # Map lowercase label to ImageType constant for weight/note lookup
        image_type_const = LABEL_TO_IMAGE_TYPE.get(image_type, ImageType.UNKNOWN)

        return ClassificationResult(
            image_type          = image_type_const,
            confidence          = confidence,
            class_probabilities = class_probs,
            weight_table        = WEIGHT_TABLES.get(image_type_const,
                                                     WEIGHT_TABLES[ImageType.UNKNOWN]),
            reliability_notes   = RELIABILITY_NOTES.get(image_type_const, []),
            method              = method,
            feature_vector      = fv,
        )

    # -----------------------------------------------------------------------
    # Internal inference
    # -----------------------------------------------------------------------

    def _hierarchical_inference(
        self, fv: FeatureVector
    ) -> tuple[str, float, dict, str]:
        """
        Run two-stage inference. Returns (image_type, confidence, probs, method).
        """
        X = fv.to_array().reshape(1, -1)

        # Stage 1 — general model
        general_probs = self._run_general(X)
        best_type     = max(general_probs, key=general_probs.get)
        confidence    = general_probs[best_type]

        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return best_type, confidence, general_probs, "ml_general"

        # Stage 2 — blend with binary models if available
        if not self._binary:
            # No binary models loaded — return general result with lower confidence
            return best_type, confidence, general_probs, "ml_general_low_conf"

        binary_probs = self._run_binary_ensemble(X)
        blended      = self._blend(general_probs, binary_probs)
        best_type    = max(blended, key=blended.get)
        confidence   = float(blended[best_type])

        return best_type, confidence, blended, "ml_ensemble"

    def _run_general(self, X: np.ndarray) -> dict[str, float]:
        """Run general 5-class model and return probability dict."""
        rf    = self._general["rf"]
        le    = self._general["le"]
        proba = rf.predict_proba(X)[0]
        return {
            str(le.classes_[i]): float(proba[i])
            for i in range(len(le.classes_))
        }

    def _run_binary_ensemble(self, X: np.ndarray) -> dict[str, float]:
        """
        Run each binary model and collect positive-class probabilities.
        Normalise to sum to 1 so they can be blended with general probs.
        """
        raw = {}
        for image_type, rf in self._binary.items():
            proba             = rf.predict_proba(X)[0]
            pos_idx           = 1 if proba.shape[0] > 1 else 0
            raw[image_type]   = float(proba[pos_idx])

        # Fill in any missing types with 0
        for t in MODEL_FILES:
            if t != "general" and t not in raw:
                raw[t] = 0.0

        # Normalise
        total = sum(raw.values())
        if total < 1e-8:
            n = len(raw)
            return {t: 1.0 / n for t in raw}
        return {t: v / total for t, v in raw.items()}

    def _blend(
        self,
        general_probs: dict[str, float],
        binary_probs:  dict[str, float],
    ) -> dict[str, float]:
        """
        Weighted blend of general and binary probability dicts.
        Keys from general_probs are used (the label encoder classes).
        Binary probs are matched by key — unmatched keys default to 0.
        """
        blended = {}
        for t in general_probs:
            g = general_probs.get(t, 0.0)
            b = binary_probs.get(t, 0.0)
            blended[t] = BLEND_GENERAL_WEIGHT * g + BLEND_BINARY_WEIGHT * b

        # Renormalise
        total = sum(blended.values())
        if total < 1e-8:
            n = len(blended)
            return {t: 1.0 / n for t in blended}
        return {t: v / total for t, v in blended.items()}

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def __repr__(self) -> str:
        if not self._is_loaded:
            return "HierarchicalClassifier(not loaded — rule-based fallback)"
        return (
            f"HierarchicalClassifier("
            f"general=loaded, "
            f"binary={self._loaded_types})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_classifier: HierarchicalClassifier | None = None


def get_classifier(models_dir: Path = DEFAULT_MODELS_DIR) -> HierarchicalClassifier:
    """
    Return the module-level singleton classifier, loading it if needed.
    This avoids reloading models on every detection request.

    Usage:
        from ml.classifier import get_classifier
        clf = get_classifier()
        result = clf.classify(image_array)
    """
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = HierarchicalClassifier(models_dir)
        _default_classifier.load()
    return _default_classifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pkl(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)