"""
Unit tests for src.predictor and predict_sign() public API.
"""
from pathlib import Path
import numpy as np
import pytest

from src.predictor import SignPredictor, predict_sign

ROOT = Path(__file__).resolve().parent.parent


def test_sign_predictor_init():
    predictor = SignPredictor()
    assert predictor.classes == sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    assert predictor.feature_mode == "norm_eng"
    assert predictor.min_confidence > 0.0


def test_predict_sign_blank_image():
    # Completely black image -> no hands detected
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = predict_sign(black_img)

    assert isinstance(res, dict)
    assert res["hand_detected"] is False
    assert res["status"] == "no_hand_detected"
    assert res["confidence"] == 0.0
    assert res["predicted_letter"] is None
    assert res["top_predictions"] == []
    assert res["processing_time_ms"] > 0
    assert res["annotated_image"] is not None


def test_predict_sign_none_image():
    res = predict_sign(None)
    assert res["hand_detected"] is False
    assert res["status"] in ("no_hand_detected", "invalid_image")


def test_predict_sign_on_real_sample():
    sample_path = ROOT / "samples" / "Indian_A.jpg"
    if not sample_path.exists():
        pytest.skip("Sample Indian_A.jpg not found")

    import cv2
    img = cv2.imread(str(sample_path))
    res = predict_sign(img)

    assert res["hand_detected"] is True
    assert res["predicted_letter"] == "A"
    assert res["confidence"] >= 0.50
    assert len(res["top_predictions"]) == 3
    assert res["top_predictions"][0][0] == "A"
    assert res["status"] == "success"
