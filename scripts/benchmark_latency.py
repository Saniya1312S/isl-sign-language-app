"""
STEP 5 - Measure real per-frame processing latency.

Uses real sample hand images resized to 640x480 (typical webcam size), measuring:
  - hand detection (MediaPipe Hands)
  - feature engineering (norm_eng)
  - scikit-learn classifier
  - PyTorch DL classifier (if checkpoint exists)
"""
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import ALPHABET_META, ALPHABET_MODEL, MODELS_DIR, RESULTS_DIR, SAMPLES_DIR  # noqa: E402
from isl.features import build_features, hands_to_arrays                                      # noqa: E402
from isl.hands import HandDetector                                                            # noqa: E402
from isl.model_io import load_bundle                                                          # noqa: E402

N_FRAMES = 120


def main():
    bundle = load_bundle(ALPHABET_MODEL, ALPHABET_META)
    if bundle is None:
        print("Production model not found. Run scripts/train_alphabet.py first.")
        return

    sample_paths = sorted(SAMPLES_DIR.glob("Indian_*.jpg"))
    if not sample_paths:
        sample_paths = sorted(SAMPLES_DIR.glob("*.jpg"))
    imgs = [cv2.resize(cv2.imread(str(p)), (640, 480)) for p in sample_paths]

    det = HandDetector(static_image_mode=False)       # video mode = tracking, like the live app
    for im in imgs[:5]:                               # warm-up
        det.detect_bgr(im)

    # Optional DL predictor
    dl_predictor = None
    dl_path = MODELS_DIR / "isl_dl_model.pth"
    if dl_path.exists():
        try:
            from isl.predictor import DLPredictor
            dl_predictor = DLPredictor(dl_path)
        except Exception:
            pass

    t_det, t_feat, t_clf, t_dl_clf, t_tot = [], [], [], [], []
    for i in range(N_FRAMES):
        im = imgs[i % len(imgs)]
        t0 = time.perf_counter()
        hands = det.detect_bgr(im)
        t1 = time.perf_counter()
        if hands:
            coords, nh, lab = hands_to_arrays(hands)
            X = build_features(coords, nh, bundle.feature_mode, lab)
            t2 = time.perf_counter()
            bundle.model.predict_proba(X)
            t3 = time.perf_counter()
            t_feat.append(t2 - t1)
            t_clf.append(t3 - t2)
            if dl_predictor:
                t_dl_0 = time.perf_counter()
                dl_predictor.predict_features(X)
                t_dl_clf.append(time.perf_counter() - t_dl_0)
        t_det.append(t1 - t0)
        t_tot.append(time.perf_counter() - t0)

    ms = lambda a, q=50: float(np.percentile(np.array(a) * 1000, q)) if a else 0.0

    stages = {
        "hand detection (MediaPipe)": round(ms(t_det), 2),
        "feature building": round(ms(t_feat), 3),
        "classifier": round(ms(t_clf), 3),
    }
    if dl_predictor and t_dl_clf:
        stages["classifier (PyTorch DL)"] = round(ms(t_dl_clf), 3)

    res = {
        "machine": f"{platform.processor() or platform.machine()} · {platform.system()} · Python {platform.python_version()}",
        "frames_measured": N_FRAMES,
        "image_size": "640x480",
        "stages_ms_p50": stages,
        "classifier_ms_p50": round(ms(t_clf), 3),
        "total_ms_p50": round(ms(t_tot), 2),
        "total_ms_p95": round(ms(t_tot, 95), 2),
        "note": ("Measured on real hand images resized to 640x480 with MediaPipe Hands tracking mode. "
                 "Network time for browser camera streaming and translation / speech is not included.")
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latency.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
