"""End-to-end checks with the REAL MediaPipe detector and the saved model (skipped if not trained yet)."""
import cv2
import numpy as np
import pytest

from isl.config import ALPHABET_META, ALPHABET_MODEL, SAMPLES_DIR
from isl.hands import HandDetector
from isl.live import SignVideoProcessor, analyse_image
from isl.model_io import load_bundle
from isl.predictor import SignPredictor

pytestmark = pytest.mark.skipif(not ALPHABET_MODEL.exists(), reason="run scripts/train_alphabet.py first")


@pytest.fixture(scope="module")
def stack():
    bundle = load_bundle(ALPHABET_MODEL, ALPHABET_META)
    return HandDetector(static_image_mode=True), SignPredictor(bundle, min_confidence=0.0), bundle


def read(name):
    return cv2.imread(str(SAMPLES_DIR / name))


def test_model_metadata_is_consistent(stack):
    _, _, bundle = stack
    assert bundle.feature_mode == "norm_eng"
    assert bundle.classes == sorted(bundle.classes) and len(bundle.classes) == 26
    assert bundle.model.named_steps["clf"].n_features_in_ == 171


def test_studio_samples_are_recognised(stack):
    det, pred, _ = stack
    letters = [p.stem.split("_")[1] for p in sorted(SAMPLES_DIR.glob("Indian_*.jpg"))]
    ok = sum(analyse_image(read(f"Indian_{L}.jpg"), det, pred)[1].label == L for L in letters)
    assert ok / len(letters) >= 0.85, f"only {ok}/{len(letters)} sample letters recognised"


def test_external_samples_run_without_error(stack):
    det, pred, _ = stack
    for p in sorted(SAMPLES_DIR.glob("External_*.jpg")):
        _, res = analyse_image(cv2.imread(str(p)), det, pred)
        assert res.reason in ("ok", "no_hand", "low_confidence")


def test_blank_image_means_no_hand(stack):
    det, pred, _ = stack
    _, res = analyse_image(np.zeros((240, 320, 3), np.uint8), det, pred)
    assert res.label is None and res.reason == "no_hand"


def test_mirrored_picture_gives_same_letter_most_of_the_time(stack):
    """Webcams are often mirrored; the new pipeline must not care (the old one collapsed to ~1%)."""
    det, pred, _ = stack
    names = sorted(p.name for p in SAMPLES_DIR.glob("Indian_*.jpg"))
    same = 0
    for n in names:
        a = analyse_image(read(n), det, pred)[1].label
        b = analyse_image(cv2.flip(read(n), 1), det, pred)[1].label
        same += (a == b)
    assert same / len(names) >= 0.8


def test_live_processor_types_a_letter_and_ignores_empty_frames(stack):
    _, _, bundle = stack
    proc = SignVideoProcessor()
    proc.configure({"alphabet": SignPredictor(bundle, 0.5)}, "alphabet", hold_frames=6, min_confidence=0.5, mirror_view=False)
    img = cv2.resize(read("Indian_L.jpg"), (640, 480))
    out = None
    for _ in range(40):
        out = proc.process_bgr(img)
    assert out.shape == img.shape
    assert proc.buffer.text.startswith("L") and len(proc.buffer.text) == 1      # typed once, not 30 times
    blank = np.zeros_like(img)
    for _ in range(15):
        proc.process_bgr(blank)
    assert proc.last.reason == "no_hand" and proc.buffer.text == "L"
    assert proc.stats()["frames"] == 55 and proc.stats()["fps"] > 0
