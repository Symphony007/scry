# ml/type_classifier.py

"""
Image type classifier for the Scry detection pipeline.

Classifies images into six types:
    Photographic, Scanned, AI-Generated, Screenshot, Synthetic, Unknown

Each classification drives the dynamic weight table selection in the
aggregator — this is the architectural fix for the Mandrill problem.

Training:
    The classifier is a Random Forest trained on extracted feature vectors.
    Because we may not always have enough real training data, the classifier
    includes a rule-based fallback that uses feature thresholds directly.
    The rule-based path is honest about its lower confidence.

The type-specific detector weight tables defined here are the single
source of truth for Phase 4. The aggregator imports them directly.
"""

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from ml.type_features import (
    extract_features, FeatureVector,
    ImageType, ALL_IMAGE_TYPES,
)
from detectors.aggregator import DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Type-specific detector weight tables
# These are the values from the project plan, Phase 4.
# Aggregator imports WEIGHT_TABLES directly.
# ---------------------------------------------------------------------------

WEIGHT_TABLES: dict[str, dict[str, float]] = {
    ImageType.PHOTOGRAPHIC: {
        "RS Analysis" : 2.0,
        "Chi-Square"  : 1.5,
        "Histogram"   : 1.0,
        "Entropy"     : 0.5,
    },
    ImageType.SCANNED: {
        "RS Analysis" : 2.0,
        "Chi-Square"  : 0.3,   # unreliable — Mandrill problem
        "Histogram"   : 0.3,   # unreliable on scanned images
        "Entropy"     : 0.8,
    },
    ImageType.AI_GENERATED: {
        "RS Analysis" : 1.5,
        "Chi-Square"  : 1.2,
        "Histogram"   : 1.0,
        "Entropy"     : 0.0,   # useless — AI images have naturally high entropy
    },
    ImageType.SCREENSHOT: {
        "RS Analysis" : 1.2,
        "Chi-Square"  : 0.5,
        "Histogram"   : 1.8,   # most reliable for structured pixel patterns
        "Entropy"     : 0.0,   # useless on screenshots
    },
    ImageType.SYNTHETIC: {
        "RS Analysis" : 0.0,
        "Chi-Square"  : 0.0,
        "Histogram"   : 0.0,
        "Entropy"     : 0.0,   # all detectors unreliable — manual review only
    },
    ImageType.UNKNOWN: {
        "RS Analysis" : 0.8,
        "Chi-Square"  : 0.8,
        "Histogram"   : 0.8,
        "Entropy"     : 0.3,
    },
}

# Reliability notes per image type — used in web interface and reports
RELIABILITY_NOTES: dict[str, list[str]] = {
    ImageType.PHOTOGRAPHIC: [
        "All detectors reliable.",
        "RS Analysis is primary signal.",
    ],
    ImageType.SCANNED: [
        "Chi-Square is unreliable — Mandrill problem applies.",
        "Histogram is unreliable on scanned images.",
        "RS Analysis is primary signal.",
        "Results have higher uncertainty than photographic images.",
    ],
    ImageType.AI_GENERATED: [
        "Entropy detector excluded — AI images have naturally high LSB entropy.",
        "RS Analysis and Chi-Square are primary signals.",
        "Detection accuracy is lower than for photographic images.",
    ],
    ImageType.SCREENSHOT: [
        "Entropy detector excluded — structured pixels produce false signals.",
        "Chi-Square has reduced weight — pair distributions are naturally structured.",
        "Histogram is primary signal.",
    ],
    ImageType.SYNTHETIC: [
        "All statistical detectors are unreliable on synthetic images.",
        "Manual review is required.",
        "No automated verdict should be trusted.",
    ],
    ImageType.UNKNOWN: [
        "Image type could not be determined with confidence.",
        "All detectors applied with reduced weight.",
        "Interpret results with caution.",
    ],
}


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """
    Output from the image type classifier.

    Attributes:
        image_type        : primary predicted image type (ImageType constant)
        confidence        : confidence in the primary prediction [0, 1]
        class_probabilities: dict mapping each ImageType to its probability
        weight_table      : detector weight dict for this image type
        reliability_notes : list of human-readable reliability warnings
        method            : "ml" if Random Forest was used, "rules" if fallback
        feature_vector    : the FeatureVector used for classification
    """
    image_type          : str
    confidence          : float
    class_probabilities : dict[str, float]
    weight_table        : dict[str, float]
    reliability_notes   : list[str]
    method              : str
    feature_vector      : FeatureVector

    def __str__(self):
        lines = [
            f"Image Type  : {self.image_type} "
            f"({self.confidence * 100:.1f}% confidence) [{self.method}]",
            "Class Probabilities:",
        ]
        for t, p in sorted(
            self.class_probabilities.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {t:<20} {p * 100:.1f}%")
        lines.append("Reliability Notes:")
        for note in self.reliability_notes:
            lines.append(f"  • {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule-based fallback classifier
# Used when no trained ML model is available.
# ---------------------------------------------------------------------------

def _classify_by_rules(fv: FeatureVector) -> tuple[str, float, dict[str, float]]:
    """
    Rule-based image type classification from feature thresholds.
    Returns (image_type, confidence, class_probabilities).

    These rules encode domain knowledge about each image type's
    feature signature. They are deliberately conservative —
    confidence is capped at 0.75 to reflect that rules are less
    reliable than a trained model.

    Rules are checked in priority order. The first matching rule wins.
    """
    scores = {t: 0.0 for t in ALL_IMAGE_TYPES}

    # --- Synthetic: uniform pixel distribution, very low noise ---
    if fv.pixel_std < 5.0 and fv.noise_variance < 2.0:
        scores[ImageType.SYNTHETIC] += 3.0

    # --- Screenshot: many histogram peaks, high edge linearity,
    #     low noise, often has gray/flat regions ---
    if fv.histogram_peaks > 15 and fv.edge_linearity > 0.6:
        scores[ImageType.SCREENSHOT] += 2.5
    if fv.smooth_region_frac > 0.5 and fv.noise_variance < 10.0:
        scores[ImageType.SCREENSHOT] += 1.0
    if fv.histogram_peaks > 20:
        scores[ImageType.SCREENSHOT] += 1.0

    # --- Scanned: high noise variance (film grain), low spatial
    #     correlation of noise (random grain), high LSB entropy ---
    if fv.noise_variance > 40.0 and fv.noise_spatial_corr < 0.15:
        scores[ImageType.SCANNED] += 2.5
    if fv.lsb_entropy > 0.92 and fv.noise_variance > 30.0:
        scores[ImageType.SCANNED] += 1.5

    # --- AI-Generated: near-perfect LSB entropy, low block boundary
    #     artifacts, smooth noise profile, broad pixel distribution ---
    if fv.lsb_entropy > 0.95 and fv.block_boundary_delta < 0.5:
        scores[ImageType.AI_GENERATED] += 2.0
    if fv.lsb_pair_balance < 0.05 and fv.pixel_std > 40.0:
        scores[ImageType.AI_GENERATED] += 1.5
    if fv.noise_spatial_corr > 0.3 and fv.lsb_entropy > 0.90:
        scores[ImageType.AI_GENERATED] += 1.0

    # --- Photographic: moderate noise, low-to-medium LSB entropy,
    #     natural gradient distribution, edge sharpness ---
    if 0.60 < fv.lsb_entropy < 0.90 and fv.noise_variance > 5.0:
        scores[ImageType.PHOTOGRAPHIC] += 2.0
    if fv.gradient_std > 10.0 and fv.edge_density > 0.05:
        scores[ImageType.PHOTOGRAPHIC] += 1.5
    if fv.pixel_std > 30.0 and fv.smooth_region_frac < 0.6:
        scores[ImageType.PHOTOGRAPHIC] += 1.0

    # If no type has a meaningful score → Unknown
    total = sum(scores.values())
    if total < 1.0:
        probs = {t: (1.0 / len(ALL_IMAGE_TYPES)) for t in ALL_IMAGE_TYPES}
        return ImageType.UNKNOWN, 0.3, probs

    # Normalize scores to probabilities
    probs       = {t: scores[t] / total for t in ALL_IMAGE_TYPES}
    best_type   = max(probs, key=probs.get)
    confidence  = min(0.75, probs[best_type])  # cap at 0.75 for rule-based

    return best_type, confidence, probs


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

class ImageTypeClassifier:
    """
    Random Forest classifier for image type prediction.

    Usage:
        clf = ImageTypeClassifier()

        # If a trained model exists:
        clf.load("data/models/type_classifier.pkl")

        # Classify an image:
        result = clf.classify(image_array)

        # Train a new model (requires labelled feature vectors):
        clf.train(X, y)
        clf.save("data/models/type_classifier.pkl")

    When no model is loaded, the classifier automatically falls back
    to rule-based classification and clearly labels results as "rules".
    """

    MODEL_VERSION    = "1.0"
    FEATURE_VERSION  = "1.0"

    def __init__(self):
        self._model          = None
        self._is_trained     = False
        self._training_info  : dict = {}

    def load(self, model_path: str) -> bool:
        """
        Load a trained model from disk.

        Returns True if loaded successfully, False if file not found
        or version mismatch. Never raises.
        """
        import pickle
        p = Path(model_path)
        if not p.exists():
            return False
        try:
            with open(p, "rb") as f:
                bundle = pickle.load(f)

            # Version check — models are not interchangeable across
            # feature engineering versions
            if bundle.get("feature_version") != self.FEATURE_VERSION:
                print(
                    f"[CLASSIFIER] Model feature version "
                    f"'{bundle.get('feature_version')}' does not match "
                    f"current feature version '{self.FEATURE_VERSION}'. "
                    f"Model not loaded — retrain required."
                )
                return False

            self._model         = bundle["model"]
            self._training_info = bundle.get("training_info", {})
            self._is_trained    = True
            print(
                f"[CLASSIFIER] Model loaded from {model_path}. "
                f"Trained on {self._training_info.get('total_samples', '?')} samples."
            )
            return True

        except Exception as e:
            print(f"[CLASSIFIER] Failed to load model: {e}")
            return False

    def save(self, model_path: str) -> None:
        """Save the trained model and its metadata to disk."""
        import pickle
        if not self._is_trained:
            raise RuntimeError("No trained model to save.")
        bundle = {
            "model"           : self._model,
            "model_version"   : self.MODEL_VERSION,
            "feature_version" : self.FEATURE_VERSION,
            "training_info"   : self._training_info,
        }
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"[CLASSIFIER] Model saved to {model_path}.")

    def train(
        self,
        X: np.ndarray,
        y: list[str],
        n_estimators: int = 200,
        test_size: float = 0.2,
    ) -> dict:
        """
        Train a Random Forest classifier.

        Args:
            X            : feature matrix (n_samples x 24)
            y            : list of ImageType label strings
            n_estimators : number of trees in the forest
            test_size    : fraction held out for validation

        Returns:
            dict with: accuracy, per_class_accuracy, feature_importances,
                       n_train, n_test, label_counts
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report

        if len(X) < 10:
            raise ValueError(
                f"Need at least 10 training samples, got {len(X)}."
            )

        le      = LabelEncoder()
        le.fit(ALL_IMAGE_TYPES)
        y_enc   = le.transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=42, stratify=y_enc
        )

        rf = RandomForestClassifier(
            n_estimators     = n_estimators,
            max_features     = "sqrt",
            min_samples_leaf = 2,
            random_state     = 42,
            n_jobs           = -1,
        )
        rf.fit(X_train, y_train)

        y_pred   = rf.predict(X_test)
        accuracy = float(accuracy_score(y_test, y_pred))

        # Per-class accuracy
        per_class = {}
        for i, label in enumerate(le.classes_):
            mask = y_test == i
            if mask.any():
                per_class[label] = float(accuracy_score(y_test[mask], y_pred[mask]))

        # Store label encoder alongside model so predict() can decode
        self._model = {"rf": rf, "le": le}
        self._is_trained = True
        self._training_info = {
            "total_samples"    : len(X),
            "n_train"          : len(X_train),
            "n_test"           : len(X_test),
            "accuracy"         : accuracy,
            "per_class_accuracy": per_class,
            "label_counts"     : {
                t: int(np.sum(np.array(y) == t)) for t in ALL_IMAGE_TYPES
            },
        }

        print(f"[CLASSIFIER] Training complete. Accuracy: {accuracy:.3f}")
        return self._training_info

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        Classify an image into one of the six image types.

        Args:
            image: RGB image array (H x W x 3, uint8)

        Returns:
            ClassificationResult with type, confidence, weights, and notes.
        """
        fv = extract_features(image)

        if self._is_trained and self._model is not None:
            image_type, confidence, class_probs = self._classify_ml(fv)
            method = "ml"
        else:
            image_type, confidence, class_probs = _classify_by_rules(fv)
            method = "rules"

        # If ML confidence is low, blend with rule-based
        if method == "ml" and confidence < 0.5:
            _, rule_conf, rule_probs = _classify_by_rules(fv)
            # Blend: 60% ML, 40% rules when confidence is low
            for t in ALL_IMAGE_TYPES:
                class_probs[t] = 0.6 * class_probs[t] + 0.4 * rule_probs[t]
            image_type = max(class_probs, key=class_probs.get)
            confidence = float(class_probs[image_type])
            method     = "ml+rules"

        weight_table     = WEIGHT_TABLES.get(image_type, WEIGHT_TABLES[ImageType.UNKNOWN])
        reliability_notes = RELIABILITY_NOTES.get(image_type, [])

        return ClassificationResult(
            image_type          = image_type,
            confidence          = confidence,
            class_probabilities = class_probs,
            weight_table        = weight_table,
            reliability_notes   = reliability_notes,
            method              = method,
            feature_vector      = fv,
        )

    def _classify_ml(
        self, fv: FeatureVector
    ) -> tuple[str, float, dict[str, float]]:
        """Run the trained Random Forest on a feature vector."""
        rf = self._model["rf"]
        le = self._model["le"]

        X          = fv.to_array().reshape(1, -1)
        proba      = rf.predict_proba(X)[0]
        class_probs = {
            le.classes_[i]: float(proba[i])
            for i in range(len(le.classes_))
        }

        best_type  = max(class_probs, key=class_probs.get)
        confidence = float(class_probs[best_type])
        return best_type, confidence, class_probs

    def get_feature_importances(self) -> dict[str, float] | None:
        """
        Return feature importances from the trained Random Forest.
        Returns None if no model is trained.
        """
        if not self._is_trained:
            return None
        from ml.type_features import FeatureVector
        rf     = self._model["rf"]
        names  = FeatureVector.feature_names()
        return {
            name: float(imp)
            for name, imp in zip(names, rf.feature_importances_)
        }