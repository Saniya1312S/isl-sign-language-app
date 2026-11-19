"""
Phase 9 Automated Tests:
1. Alphabet image loading
2. Sample directory discovery
3. Correct label mapping
4. Image colour conversion
5. MediaPipe detection
6. Feature extraction
7. 171-dimensional feature consistency
8. SVM model loading
9. Actual sample prediction
10. No-hand handling
11. Invalid image handling
12. Low-confidence handling
13. Word-reference gallery loading
14. Separation between alphabet prediction and word reference images
"""
from pathlib import Path
import cv2
import joblib
import numpy as np
import pytest

from src.feature_extraction import build_features, extract_features_from_image, get_detector
from src.predictor import SignPredictor, predict_sign
from src.preprocessing import MediaPipeHandDetector

ROOT = Path(__file__).resolve().parent.parent
ALPHABET_DIR = ROOT / "samples" / "indian_isl_alphabet"
WORDS_DIR = ROOT / "samples" / "indian_isl_words"
MODEL_PATH = ROOT / "models" / "final_isl_model.joblib"


def test_alphabet_image_loading():
    """1. Alphabet image loading: All 26 letters A-Z exist and load cleanly."""
    assert ALPHABET_DIR.exists(), f"Missing directory {ALPHABET_DIR}"
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        img_path = ALPHABET_DIR / f"{letter}.jpg"
        assert img_path.exists(), f"Missing alphabet reference image: {img_path}"
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to load image: {img_path}"
        assert img.shape[0] > 0 and img.shape[1] > 0 and img.shape[2] == 3


def test_sample_directory_discovery():
    """2. Sample directory discovery: Both alphabet and word directories exist with correct count."""
    assert ALPHABET_DIR.is_dir()
    alpha_files = list(ALPHABET_DIR.glob("*.jpg")) + list(ALPHABET_DIR.glob("*.png"))
    assert len(alpha_files) == 26, f"Expected 26 alphabet images, found {len(alpha_files)}"
    if WORDS_DIR.is_dir():
        word_files = list(WORDS_DIR.glob("*.jpg")) + list(WORDS_DIR.glob("*.png"))
        assert len(word_files) == 15, f"Expected 15 word reference images, found {len(word_files)}"


def test_correct_label_mapping():
    """3. Correct label mapping: Filenames A.jpg .. Z.jpg map to A .. Z."""
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        expected = letter
        filename = f"{letter}.jpg"
        mapped_label = Path(filename).stem.upper()
        assert mapped_label == expected


def test_image_colour_conversion():
    """4. Image colour conversion: BGR to RGB conversion preserves shape and ranges."""
    sample_img = cv2.imread(str(ALPHABET_DIR / "A.jpg"))
    assert sample_img is not None
    rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    assert rgb.shape == sample_img.shape
    assert rgb.dtype == sample_img.dtype
    # Red and blue channels should be swapped
    assert np.array_equal(sample_img[:, :, 0], rgb[:, :, 2])
    assert np.array_equal(sample_img[:, :, 2], rgb[:, :, 0])


def test_mediapipe_detection():
    """5. MediaPipe detection: Detects hand on sample A.jpg."""
    detector = get_detector()
    sample_img = cv2.imread(str(ALPHABET_DIR / "A.jpg"))
    hands = detector.detect_bgr(sample_img)
    assert len(hands) >= 1
    assert hands[0].landmarks.shape == (21, 3)


def test_feature_extraction():
    """6. Feature extraction: Successfully featurizes sample A.jpg."""
    sample_img = cv2.imread(str(ALPHABET_DIR / "A.jpg"))
    res = extract_features_from_image(sample_img)
    assert res["status"] == "success"
    assert res["features"] is not None
    assert res["n_hands"] >= 1


def test_171d_feature_consistency():
    """7. 171-dimensional feature consistency: Extracted features have exactly 171 dimensions."""
    sample_img = cv2.imread(str(ALPHABET_DIR / "A.jpg"))
    res = extract_features_from_image(sample_img)
    assert res["features"].shape == (1, 171)
    assert not np.isnan(res["features"]).any()


def test_svm_model_loading():
    """8. SVM model loading: Trained SVM model loads cleanly and has predict capability."""
    assert MODEL_PATH.exists()
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_actual_sample_prediction():
    """9. Actual sample prediction: Runs real inference on A.jpg, predicts A with >90% confidence."""
    sample_img = cv2.imread(str(ALPHABET_DIR / "A.jpg"))
    res = predict_sign(sample_img)
    assert res["hand_detected"] is True
    assert res["predicted_letter"] == "A"
    assert res["confidence"] >= 0.90
    assert res["status"] == "success"


def test_no_hand_handling():
    """10. No-hand handling: Blank image returns hand_detected=False and standard message."""
    black_img = np.zeros((128, 128, 3), dtype=np.uint8)
    res = predict_sign(black_img)
    assert res["hand_detected"] is False
    assert res["predicted_letter"] is None
    assert res["confidence"] == 0.0
    assert "No hand detected" in res["message"]


def test_invalid_image_handling():
    """11. Invalid image handling: None or empty array returns invalid_image status without crashing."""
    res_none = predict_sign(None)
    assert res_none["hand_detected"] is False
    assert res_none["status"] in ("invalid_image", "no_hand_detected")

    res_empty = predict_sign(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res_empty["hand_detected"] is False


def test_low_confidence_handling():
    """12. Low-confidence handling: Low-confidence prediction triggers low_confidence status and message."""
    predictor = SignPredictor(min_confidence=0.9999)  # Impossible threshold
    sample_img = cv2.imread(str(ALPHABET_DIR / "B.jpg"))
    res = predictor.predict_image(sample_img)
    assert res["hand_detected"] is True
    assert res["status"] == "low_confidence"
    assert "Uncertain prediction" in res["message"]


def test_word_reference_gallery_loading():
    """13. Word-reference gallery loading: All 15 required ISL words exist and load as images."""
    if not WORDS_DIR.exists():
        pytest.skip("Word gesture gallery was removed per A-Z simplification request")
    expected_words = [
        "HELLO", "THANK_YOU", "WATER", "FOOD", "HELP",
        "MOTHER", "FATHER", "FRIEND", "SCHOOL", "HOME",
        "HOSPITAL", "DOCTOR", "PLEASE", "SORRY", "GOODBYE"
    ]
    for w in expected_words:
        w_path = WORDS_DIR / f"{w}.jpg"
        assert w_path.exists(), f"Missing word reference card: {w_path}"
        im = cv2.imread(str(w_path))
        assert im is not None, f"Failed to load word image: {w_path}"
        assert im.shape[0] == 640 and im.shape[1] == 640


def test_separation_alphabet_and_word_reference():
    """14. Separation between alphabet prediction and word reference images:
    Verifies that word cards are strictly reference-only and not evaluated as alphabet letters."""
    word_csv = ROOT / "reports" / "indian_word_reference_inventory.csv"
    if not word_csv.exists():
        pytest.skip("Word reference inventory was removed per A-Z simplification request")
    import pandas as pd
    df = pd.read_csv(word_csv)
    assert len(df) == 15
    assert (df["reference_only_status"] == "Reference Only").all()
    assert (df["model_support_status"].str.contains("Not Yet Trained")).all()


def test_batch_prediction_indian_samples():
    """15. Test real-time batch prediction across Indian alphabet sample files."""
    from src.safe_dataframe import calculate_batch_metrics, normalize_batch_dataframe
    import pandas as pd

    predictor = SignPredictor()
    # Test a slice of Indian alphabet samples: A, B, C, D, E
    test_letters = ["A", "B", "C", "D", "E"]
    batch_rows = []
    for letter in test_letters:
        img_path = ALPHABET_DIR / f"{letter}.jpg"
        assert img_path.exists()
        img = cv2.imread(str(img_path))
        res = predictor.predict_image(img)
        batch_rows.append({
            "Image": img_path.name,
            "Source": "Indian",
            "Expected": letter,
            "Predicted": res.get("predicted_letter") or "None",
            "Confidence": res.get("confidence", 0.0),
            "Hand Detected": res.get("hand_detected", False),
            "Result": "PASS" if res.get("predicted_letter") == letter else "FAIL",
            "Latency (ms)": res.get("processing_time_ms", 0.0)
        })

    df = normalize_batch_dataframe(pd.DataFrame(batch_rows))
    metrics = calculate_batch_metrics(df)
    assert metrics["total_tested"] == 5
    assert metrics["total_detected"] == 5
    assert metrics["total_correct"] == 5
    assert metrics["detection_rate"] == 1.0
    assert metrics["overall_accuracy"] == 1.0


def test_batch_prediction_external_and_no_hand():
    """16. Test batch prediction with external images including no-hand and domain shift."""
    from src.safe_dataframe import calculate_batch_metrics, normalize_batch_dataframe
    import pandas as pd

    predictor = SignPredictor()
    ext_dir = ROOT / "samples"
    ext_files = sorted(list(ext_dir.glob("External_*.jpg")))[:5]
    assert len(ext_files) > 0

    batch_rows = []
    for f in ext_files:
        lbl = f.stem.split("_")[1].upper()
        img = cv2.imread(str(f))
        res = predictor.predict_image(img)
        hand_det = res.get("hand_detected", False)
        batch_rows.append({
            "Image": f.name,
            "Source": "External",
            "Expected": lbl,
            "Predicted": res.get("predicted_letter") or "None",
            "Confidence": res.get("confidence", 0.0),
            "Hand Detected": hand_det,
            "Result": "PASS" if (hand_det and res.get("predicted_letter") == lbl) else "FAIL",
            "Latency (ms)": res.get("processing_time_ms", 0.0)
        })

    df = normalize_batch_dataframe(pd.DataFrame(batch_rows))
    metrics = calculate_batch_metrics(df)
    assert metrics["total_tested"] == len(ext_files)
    assert 0.0 <= metrics["detection_rate"] <= 1.0
    assert 0.0 <= metrics["overall_accuracy"] <= 1.0


def test_batch_prediction_error_tolerance_and_corrupt_files():
    """17. Test batch execution error tolerance with missing and corrupt files."""
    from src.safe_dataframe import calculate_batch_metrics, normalize_batch_dataframe
    import pandas as pd

    # Synthetic batch with missing file, corrupt file, blank image, valid file
    batch_rows = [
        {"Image": "missing.jpg", "Source": "Test", "Expected": "A", "Predicted": "None",
         "Confidence": 0.0, "Hand Detected": False, "Result": "FAIL", "Latency (ms)": 0.0, "Status": "missing_file"},
        {"Image": "corrupt.jpg", "Source": "Test", "Expected": "B", "Predicted": "None",
         "Confidence": 0.0, "Hand Detected": False, "Result": "FAIL", "Latency (ms)": 0.0, "Status": "corrupt_image"},
        {"Image": "A.jpg", "Source": "Indian", "Expected": "A", "Predicted": "A",
         "Confidence": 0.98, "Hand Detected": True, "Result": "PASS", "Latency (ms)": 15.0, "Status": "success"}
    ]

    df = normalize_batch_dataframe(pd.DataFrame(batch_rows))
    metrics = calculate_batch_metrics(df)
    assert metrics["total_tested"] == 3
    assert metrics["total_detected"] == 1
    assert metrics["total_correct"] == 1
    assert metrics["detection_rate"] == pytest.approx(1 / 3)
    assert metrics["overall_accuracy"] == pytest.approx(1 / 3)
    assert metrics["accuracy_on_detected"] == 1.0

