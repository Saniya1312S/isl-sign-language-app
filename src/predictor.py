"""
Phase 10: Reusable SignPredictor & predict_sign(image) public API.
Loads the final selected ISL model and executes identical preprocessing and feature extraction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np

from .feature_extraction import extract_features_from_image, get_detector
from .preprocessing import MediaPipeHandDetector

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "final_isl_model.joblib"
DEFAULT_META_PATH = ROOT / "models" / "model_metadata.json"


class SignPredictor:
    """Production SignPredictor wrapping the trained final ISL model."""

    def __init__(self, model_path: Path | str | None = None,
                 meta_path: Path | str | None = None,
                 min_confidence: float = 0.50):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.meta_path = Path(meta_path) if meta_path else DEFAULT_META_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found at: {self.model_path}. Run python -m src.train_models first.")

        self.model = joblib.load(self.model_path)

        if self.meta_path.exists():
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.classes = self.metadata.get("class_labels", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            self.feature_mode = self.metadata.get("feature_mode", "norm_eng")
            self.min_confidence = self.metadata.get("default_confidence_threshold", min_confidence)
        else:
            self.metadata = {}
            self.classes = list(getattr(self.model, "classes_", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")))
            self.feature_mode = "norm_eng"
            self.min_confidence = min_confidence

    def predict_image(self, image: np.ndarray,
                      detector: Optional[MediaPipeHandDetector] = None) -> Dict[str, Any]:
        """
        Run end-to-end detection, landmark extraction, and prediction on an image.
        """
        t0 = time.perf_counter()

        feat_res = extract_features_from_image(image, detector=detector, mode=self.feature_mode)
        status = feat_res["status"]
        annotated_image = feat_res["annotated_image"]

        if status != "success" or feat_res["features"] is None:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "predicted_letter": None,
                "confidence": 0.0,
                "top_predictions": [],
                "hand_detected": False,
                "processing_time_ms": round(elapsed_ms, 2),
                "annotated_image": annotated_image,
                "status": "no_hand_detected" if status == "no_hand_detected" else "invalid_image"
            }

        X = feat_res["features"]

        # Prediction probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
        else:
            # Fallback for models without predict_proba (decision function / uniform)
            if hasattr(self.model, "decision_function"):
                df = self.model.decision_function(X)[0]
                exp_df = np.exp(df - np.max(df))
                probs = exp_df / exp_df.sum()
            else:
                pred_label = self.model.predict(X)[0]
                probs = np.zeros(len(self.classes))
                if pred_label in self.classes:
                    probs[self.classes.index(pred_label)] = 1.0

        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions: List[Tuple[str, float]] = [
            (str(self.classes[i]), float(round(probs[i], 4))) for i in top_indices
        ]

        best_label, best_conf = top_predictions[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred_status = "success"
        if best_conf < self.min_confidence:
            pred_status = "low_confidence"

        return {
            "predicted_letter": best_label if pred_status == "success" else best_label,
            "confidence": best_conf,
            "top_predictions": top_predictions,
            "hand_detected": True,
            "processing_time_ms": round(elapsed_ms, 2),
            "annotated_image": annotated_image,
            "status": pred_status
        }


# Singleton predictor cache
_GLOBAL_PREDICTOR: Optional[SignPredictor] = None


def predict_sign(image: np.ndarray, model_path: Path | str | None = None) -> Dict[str, Any]:
    """
    Mandatory Phase 10 Single Reusable Prediction Function.

    Usage:
        result = predict_sign(image)
        print(result["predicted_letter"], result["confidence"])
    """
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None or model_path is not None:
        _GLOBAL_PREDICTOR = SignPredictor(model_path=model_path)
    return _GLOBAL_PREDICTOR.predict_image(image)
