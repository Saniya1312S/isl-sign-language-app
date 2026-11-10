"""
STEP 4 - Compare the ORIGINAL project's model with the new one under realistic conditions.

The original alphabet model (a RandomForest on raw landmarks, hand slot chosen by MediaPipe's Left/Right
label) is loaded from `models/legacy/` and tested on the same landmark table in three situations:

  as_trained      images unflipped, Left -> first slot     (the convention it was trained with)
  slots_swapped   same images, Left/Right slots swapped    (what happens if the label convention differs)
  mirrored_frame  image flipped horizontally AND labels swapped  = what MediaPipe reports for a mirrored
                  webcam frame. The original Streamlit app flips the live frame (cv2.flip) before prediction.

The new model needs no label and was trained with mirror augmentation, so all situations look the same to it.
Usage: python scripts/evaluate_legacy.py [--legacy-model PATH --legacy-encoder PATH]
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.base import clone                                          # noqa: E402
from isl.config import MODELS_DIR, RESULTS_DIR                          # noqa: E402
from isl.data import load_landmarks, to_arrays                         # noqa: E402
from isl.features import build_features, mirror                        # noqa: E402
from train_alphabet import STUDY_CAP, block_split, cap_per_class, featurize, models   # noqa: E402

warnings.filterwarnings("ignore")


def swap_labels(labels):
    sw = {"Left": "Right", "Right": "Left", "": ""}
    return np.vectorize(lambda v: sw[v], otypes=[object])(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-model", default=str(MODELS_DIR / "legacy" / "alphabet_model_original.pkl"))
    ap.add_argument("--legacy-encoder", default=str(MODELS_DIR / "legacy" / "alphabet_label_encoder_original.pkl"))
    args = ap.parse_args()
    legacy, enc = joblib.load(args.legacy_model), joblib.load(args.legacy_encoder)
    # The NEW model is the *evaluation copy* (trained on studio training blocks only, never shown the external
    # images) - NOT the production model, which also saw the external images.
    sel = json.loads((RESULTS_DIR / "model_comparison.json").read_text())["selected_model"]
    full = load_landmarks()
    indian_d = full[(full.source == "Indian") & (full.n_hands > 0)].reset_index(drop=True)
    tr, _ = block_split(cap_per_class(indian_d, STUDY_CAP).reset_index(drop=True))
    new = clone(models()[sel]).fit(*featurize(tr, "norm_eng", augment=True))
    print(f"new model = evaluation copy of: {sel}", flush=True)

    df = load_landmarks()
    df = df[df.n_hands > 0]
    rows = []
    for dataset, name in (("Indian", "Indian dataset (partly seen in training)"), ("ISL_Dataset_2", "External dataset (unseen)")):
        d = df[df.source == dataset]
        a = to_arrays(d)
        known = np.isin(a["y"], enc.classes_)
        c, nh, lab, y = a["coords"][known], a["n_hands"][known], a["labels"][known], a["y"][known]
        yi = enc.transform(y)
        mc = mirror(c, nh)                                   # horizontally flipped hands
        scen = {
            "as_trained": (c, lab),
            "slots_swapped": (c, swap_labels(lab)),
            "mirrored_frame": (mc, swap_labels(lab)),
        }
        for sc, (cc, ll) in scen.items():
            X = build_features(cc, nh, "legacy_raw126", ll)
            rows.append({"model": "Original (raw + Left/Right slots)", "dataset": name, "scenario": sc,
                         "accuracy": round(float((legacy.predict(X) == yi).mean()), 4), "n": int(len(y))})
            Xn = build_features(cc, nh, "norm_eng")
            rows.append({"model": "New (normalised + mirror-aug)", "dataset": name, "scenario": sc,
                         "accuracy": round(float((new.predict(Xn) == y).mean()), 4), "n": int(len(y))})
        print(dataset, "done", flush=True)
    note = ("Original model = your RandomForest trained on 6,500 landmark rows; which images it was trained on is not recorded, so "
            "its Indian-dataset numbers are optimistic and it may also have seen some external images (its external score is "
            "therefore not guaranteed to be 'unseen'). New model = the evaluation copy that never saw the external data. "
            "'mirrored_frame' reproduces what the original Streamlit live loop feeds its model (flipped frame, Left->first slot). "
            "The robust, unambiguous finding is the collapse of the original features when the left/right convention changes.")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "legacy_comparison.json").write_text(json.dumps({"note": note, "rows": rows}, indent=1), encoding="utf-8")
    for r in rows:
        print(f"{r['model'][:12]:12s} {r['dataset'][:14]:14s} {r['scenario']:15s} {r['accuracy']:.3f} (n={r['n']})")


if __name__ == "__main__":
    main()
