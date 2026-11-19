"""
Automated Integration and API Tests for FastAPI Full-Stack ISL Website.
Tests all endpoint handler functions directly to ensure reliable, high-speed execution
across platforms without threadpool or blocking portal deadlocks.
"""
from pathlib import Path
import pytest
import cv2
import numpy as np

from backend.main import health_check, serve_index
from backend.api.recognition import (
    list_alphabet_samples,
    get_sample_image_and_prediction,
    predict_sign_endpoint,
    PredictRequest
)
from backend.api.translation import (
    translate_endpoint,
    speak_endpoint,
    TranslateRequest,
    SpeakRequest
)
from backend.api.gemini import (
    gemini_status_endpoint,
    gemini_chat_endpoint,
    gemini_validate_endpoint,
    ChatRequest,
    ValidateKeyRequest
)
from backend.services.predictor_service import encode_image_base64

ROOT = Path(__file__).resolve().parent.parent
ALPHABET_DIR = ROOT / "samples" / "indian_isl_alphabet"


def test_health_check_endpoint():
    """Verify health_check() returns 200 payload with 26 classes and 171 features."""
    data = health_check()
    assert data["status"] == "healthy"
    assert data["classes_count"] == 26
    assert data["feature_dimensions"] == 171
    assert data["two_hand_supported"] is True


def test_frontend_static_serving():
    """Verify serve_index() returns the index.html FileResponse."""
    resp = serve_index()
    assert resp.status_code == 200
    assert "index.html" in str(resp.path)


def test_samples_endpoints():
    """Verify list_alphabet_samples() and get_sample_image_and_prediction('A')."""
    data = list_alphabet_samples()
    assert data["total"] == 26
    assert len(data["samples"]) == 26

    # Test single sample evaluation for 'A'
    data_a = get_sample_image_and_prediction("A")
    assert data_a["letter"] == "A"
    assert data_a["prediction"]["hand_detected"] is True
    assert data_a["prediction"]["predicted_letter"] == "A"


def test_predict_base64_endpoint():
    """Verify predict_sign_endpoint with authentic sample image B."""
    img_path = ALPHABET_DIR / "B.jpg"
    assert img_path.exists()
    img = cv2.imread(str(img_path))
    assert img is not None

    b64_url = encode_image_base64(img)
    req = PredictRequest(image=b64_url, source="test_b")
    data = predict_sign_endpoint(req)
    assert data["hand_detected"] is True
    assert data["predicted_letter"] == "B"
    assert data["confidence"] > 0.70
    assert len(data["annotated_image"]) > 0


def test_predict_no_hand_endpoint():
    """Verify predict_sign_endpoint handles blank black image gracefully."""
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    b64_url = encode_image_base64(blank)
    req = PredictRequest(image=b64_url, source="test_blank")
    data = predict_sign_endpoint(req)
    assert data["hand_detected"] is False
    assert data["predicted_letter"] is None
    assert "No hand detected" in data["message"]


def test_translation_endpoints():
    """Verify translate_endpoint for Hindi and Marathi."""
    # Hindi
    req_hi = TranslateRequest(text="hello", target_language="Hindi")
    data_hi = translate_endpoint(req_hi)
    assert data_hi["translated_text"] == "नमस्ते"

    # Marathi
    req_mr = TranslateRequest(text="hello", target_language="Marathi")
    data_mr = translate_endpoint(req_mr)
    assert data_mr["translated_text"] == "नमस्कार"


def test_speak_endpoint():
    """Verify speak_endpoint returns speech synthesis metadata."""
    req = SpeakRequest(text="नमस्ते", language="Hindi")
    data = speak_endpoint(req)
    assert data["status"] == "success"
    assert data["browser_locale"] == "hi-IN"


def test_gemini_status_and_chat():
    """Verify Gemini diagnostic status and graceful response when key is missing."""
    # Status
    status = gemini_status_endpoint()
    assert status["sdk_available"] is True
    assert "gemini-2.0-flash" in status["configured_models"]

    # Chat without key
    chat_req = ChatRequest(message="HELLO", language="English", api_key="")
    chat_resp = gemini_chat_endpoint(chat_req)
    assert chat_resp["status"] == "no_key"
    assert "Gemini API key is not configured" in chat_resp["reply"]
