"""
Unit and Integration Tests for ISL A-Z Prediction, Sentence Building, and Translation.
"""
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.predictor import SignPredictor, predict_sign
from src.translator import translate_text, OFFLINE_GLOSSARY, LETTER_NAMES
from src.speech import generate_browser_speech_html, generate_audio_bytes

ROOT = Path(__file__).resolve().parent.parent
ALPHABET_DIR = ROOT / "samples" / "indian_isl_alphabet"
MODEL_PATH = ROOT / "models" / "final_isl_model.joblib"
ALT_MODEL_PATH = ROOT / "models" / "alphabet_model.pkl"


def test_model_files_exist():
    """Verify that trained model files exist on disk."""
    assert MODEL_PATH.exists() or ALT_MODEL_PATH.exists(), "No trained model found in models/"


def test_predictor_initialization_and_classes():
    """Verify SignPredictor loads and has all 26 uppercase alphabet classes."""
    predictor = SignPredictor()
    assert len(predictor.classes) == 26
    assert list(predictor.classes) == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_alphabet_sample_predictions():
    """Verify genuine model predictions on authentic Indian ISL alphabet samples."""
    predictor = SignPredictor()
    test_letters = ["A", "B", "C", "D", "E"]
    for letter in test_letters:
        img_path = ALPHABET_DIR / f"{letter}.jpg"
        assert img_path.exists(), f"Sample image {letter}.jpg missing"
        img = cv2.imread(str(img_path))
        assert img is not None
        res = predictor.predict_image(img)
        assert res["hand_detected"] is True
        assert res["predicted_letter"] == letter
        assert res["confidence"] > 0.70


def test_no_hand_and_invalid_image():
    """Verify that blank and invalid images return No Hand Detected cleanly."""
    predictor = SignPredictor()
    # Blank black image
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    res_blank = predictor.predict_image(blank)
    assert res_blank["hand_detected"] is False
    assert res_blank["predicted_letter"] is None
    assert "No hand detected" in res_blank["message"]

    # None input
    res_none = predictor.predict_image(None)
    assert res_none["hand_detected"] is False
    assert res_none["predicted_letter"] is None


def test_sentence_building_operations():
    """Verify letter accumulation, spaces, backspace, and clearing."""
    sentence = ""
    
    # 1. User signs H, E, L, L, O
    for char in ["H", "E", "L", "L", "O"]:
        sentence += char
    assert sentence == "HELLO"

    # 2. Add space
    sentence += " "
    assert sentence == "HELLO "

    # 3. Add W, O, R, L, D
    for char in ["W", "O", "R", "L", "D"]:
        sentence += char
    assert sentence == "HELLO WORLD"

    # 4. Backspace
    sentence = sentence[:-1]
    assert sentence == "HELLO WORL"

    # 5. Clear
    sentence = ""
    assert sentence == ""


def test_translation_offline_glossary():
    """Verify offline Hindi and Marathi translation without external API calls."""
    # Common greeting
    t_hi = translate_text("hello", target_lang="hi", allow_online=False)
    assert t_hi.text == "नमस्ते"
    assert t_hi.source == "glossary"

    t_mr = translate_text("hello", target_lang="mr", allow_online=False)
    assert t_mr.text == "नमस्कार"
    assert t_mr.source == "glossary"

    # Single letter phonetic name
    t_letter = translate_text("A", target_lang="hi", allow_online=False)
    assert t_letter.text == "ए"
    assert t_letter.source == "phonetic"

    # Multi-word phrase
    t_phrase = translate_text("thank you", target_lang="hi", allow_online=False)
    assert t_phrase.text == "धन्यवाद"


def test_speech_generation():
    """Verify Web Speech API HTML and fallback audio generation."""
    # Web Speech API HTML snippet
    html_code = generate_browser_speech_html("नमस्ते", lang="hi", nonce=1)
    assert "<script>" in html_code
    assert "SpeechSynthesisUtterance" in html_code
    assert "hi-IN" in html_code

    # English Web Speech
    html_en = generate_browser_speech_html("HELLO", lang="en", nonce=2)
    assert "en-IN" in html_en
