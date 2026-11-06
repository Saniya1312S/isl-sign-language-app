"""
STEP 5 - Measure how fast one frame is processed (the report promised latency numbers but never measured them).

Uses the sample images upscaled to 640x480 (typical webcam size) and the saved production model.
Run:  python scripts/benchmark_latency.py        (numbers depend on your CPU - quote your own machine!)
"""
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import ALPHABET_META, ALPHABET_MODEL, RESULTS_DIR, SAMPLES_DIR      # noqa: E402
from isl.features import build_features, hands_to_arrays                              # noqa: E402
from isl.hands import HandDetector                                                    # noqa: E402
from isl.model_io import load_bundle                                                  # noqa: E402

N_FRAMES = 150


def main():
    bundle = load_bundle(ALPHABET_MODEL, ALPHABET_META)
    imgs = [cv2.resize(cv2.imread(str(p)), (640, 480)) for p in sorted(SAMPLES_DIR.glob("Indian_*.jpg"))]
    det = HandDetector(static_image_mode=False)       # video mode = tracking, like the live app
    for im in imgs[:5]:                               # warm-up
        det.detect_bgr(im)
    t_det, t_feat, t_clf, t_tot = [], [], [], []
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
        t_det.append(t1 - t0)
        t_tot.append(time.perf_counter() - t0)
    ms = lambda a, q=50: float(np.percentile(np.array(a) * 1000, q))
    res = {"machine": f"{platform.processor() or platform.machine()} · {platform.system()} · Python {platform.python_version()}",
           "frames_measured": N_FRAMES, "image_size": "640x480",
           "stages_ms_p50": {"hand detection (MediaPipe)": round(ms(t_det), 2), "feature building": round(ms(t_feat), 3),
                             "classifier": round(ms(t_clf), 3)},
           "classifier_ms_p50": round(ms(t_clf), 3), "total_ms_p50": round(ms(t_tot), 2), "total_ms_p95": round(ms(t_tot, 95), 2),
           "note": ("Measured on the machine named above with a single process; your laptop will differ. Network time for the "
                    "browser camera stream and for translation / speech is NOT included.")}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "latency.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
