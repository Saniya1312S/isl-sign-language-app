"""
Predictor Service wrapping the ML pipeline for FastAPI.
Handles Base64 image decoding, MediaPipe feature extraction, SVM prediction,
and annotated image serialization.
"""
from __future__ import annotations

import base64
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to sys.path so we can import src modules
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predictor import SignPredictor
from backend.config import MODEL_PATH, ALT_MODEL_PATH, META_PATH

_PREDICTOR_INSTANCE: Optional[SignPredictor] = None


def get_predictor() -> SignPredictor:
    """Singleton getter for SignPredictor."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        model_file = MODEL_PATH if MODEL_PATH.exists() else ALT_MODEL_PATH
        _PREDICTOR_INSTANCE = SignPredictor(model_path=model_file, meta_path=META_PATH)
    return _PREDICTOR_INSTANCE


def decode_image_base64(image_data: str) -> Optional[np.ndarray]:
    """Decode a base64 string or data URL (data:image/jpeg;base64,...) to BGR image."""
    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as ex:
        print(f"Error decoding base64 image: {ex}")
        return None


def encode_image_base64(img_bgr: np.ndarray, format_ext: str = ".jpg") -> str:
    """Encode OpenCV BGR image to base64 data URL."""
    try:
        success, buffer = cv2.imencode(format_ext, img_bgr)
        if not success:
            return ""
        b64_str = base64.b64encode(buffer).decode("utf-8")
        mime = "image/jpeg" if format_ext.lower() in [".jpg", ".jpeg"] else "image/png"
        return f"data:{mime};base64,{b64_str}"
    except Exception as ex:
        print(f"Error encoding image to base64: {ex}")
        return ""


def run_prediction_on_image(image_bgr: np.ndarray) -> Dict[str, Any]:
    """Execute prediction on OpenCV BGR image and package results."""
    predictor = get_predictor()
    raw_res = predictor.predict_image(image_bgr)

    annotated_bgr = raw_res.get("annotated_image")
    annotated_b64 = encode_image_base64(annotated_bgr) if annotated_bgr is not None else ""

    # Check number of hands from raw_res if available
    n_hands = 0
    if raw_res.get("hand_detected"):
        n_hands = raw_res.get("n_hands", 1)

    return {
        "predicted_letter": raw_res.get("predicted_letter"),
        "confidence": float(raw_res.get("confidence", 0.0)),
        "hand_detected": bool(raw_res.get("hand_detected", False)),
        "n_hands": n_hands,
        "processing_time_ms": float(raw_res.get("processing_time_ms", 0.0)),
        "top_predictions": raw_res.get("top_predictions", []),
        "status": raw_res.get("status", "unknown"),
        "message": raw_res.get("message", ""),
        "annotated_image": annotated_b64
    }
