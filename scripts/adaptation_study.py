"""
STEP 3b - Few-shot personalisation study.

Finding from the main study: a model trained only on the studio dataset scores ~25% on a different dataset
(other people / backgrounds). Question: how many examples of the NEW setting does it need to recover?

Protocol (conservative, no leakage between "teach" and "test" images):
  * the external dataset is split by FILE ORDER per letter: the first `k` images are the "calibration" samples
    (what a user would record), the LAST 50% of the letter's images are the test set. Neighbouring files are
    never shared between the two sets.
  * calibration samples are added to the studio training data (mirror-augmented, weighted up),
    the model is retrained, then tested on the held-out external images AND on the studio block-test set
    (to check the model does not forget the studio data).

Writes results/adaptation.json.   Run: python scripts/adaptation_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import RANDOM_STATE, RESULTS_DIR                          # noqa: E402
from train_alphabet import STUDY_CAP, block_split, cap_per_class, featurize, load, models   # noqa: E402

WEIGHT = 15.0          # one calibration sample counts as 15 studio samples (there are ~600 studio samples per letter)
KS = (0, 1, 2, 3, 5, 8)


def selected_model():
    """The model type chosen by the main study (read from results/model_comparison.json)."""
    name = json.loads((RESULTS_DIR / "model_comparison.json").read_text())["selected_model"]
    return name, models()[name]


def main():
    _, indian, ext, ext_all = load()
    study = cap_per_class(indian, STUDY_CAP).reset_index(drop=True)
    tr, te = block_split(study)
    Xs, ys = featurize(tr, "norm_eng", augment=True)
    Xs_te, ys_te = featurize(te, "norm_eng")

    ext = ext.sort_values(["label", "frame_idx"]).reset_index(drop=True)
    test_parts, teach_pool = [], []
    for _, g in ext.groupby("label"):
        half = len(g) // 2
        teach_pool.append(g.iloc[:half])          # candidates for calibration (first half, file order)
        test_parts.append(g.iloc[half:])          # held-out test (second half)
    pool, test = pd.concat(teach_pool), pd.concat(test_parts)
    Xt, yt = featurize(test, "norm_eng")
    keep = np.isin(yt, np.unique(ys))
    Xt, yt = Xt[keep], yt[keep]
    print(f"external test images: {len(yt)} (letters: {len(set(yt))})", flush=True)

    name, base = selected_model()
    print(f"model type: {name}", flush=True)
    rows = []
    for k in KS:
        if k == 0:
            X, y, w = Xs, ys, np.ones(len(ys))
        else:
            cal = pd.concat([g.iloc[:k] for _, g in pool.groupby("label")])
            Xc, yc = featurize(cal, "norm_eng", augment=True)
            X, y = np.vstack([Xs, Xc]), np.concatenate([ys, yc])
            w = np.concatenate([np.ones(len(ys)), np.full(len(yc), WEIGHT)])
        accs, studio = [], []
        for seed in range(3):                    # 3 repeats (only matters for models with randomness)
            m = clone(base)
            if "random_state" in m.named_steps["clf"].get_params():
                m.set_params(clf__random_state=RANDOM_STATE + seed)
            m.fit(X, y, clf__sample_weight=w)
            accs.append(accuracy_score(yt, m.predict(Xt)))
            studio.append(accuracy_score(ys_te, m.predict(Xs_te)))
        rows.append({"calibration_images_per_letter": k, "external_accuracy": round(float(np.mean(accs)), 4),
                     "external_accuracy_std": round(float(np.std(accs)), 4),
                     "studio_block_accuracy": round(float(np.mean(studio)), 4)})
        print(f"  k={k}: external {rows[-1]['external_accuracy']:.3f} ± {rows[-1]['external_accuracy_std']:.3f} | "
              f"studio {rows[-1]['studio_block_accuracy']:.3f}", flush=True)
    # how much should calibration images count? (all first-half external images are used here)
    wrows = []
    cal = pool
    Xc, yc = featurize(cal, "norm_eng", augment=True)
    for w_ in (1, 5, 10, 20, 40):
        X = np.vstack([Xs, Xc]); y = np.concatenate([ys, yc])
        ww = np.concatenate([np.ones(len(ys)), np.full(len(yc), float(w_))])
        m = clone(base).fit(X, y, clf__sample_weight=ww)
        wrows.append({"weight": w_, "external_accuracy": round(float(accuracy_score(yt, m.predict(Xt))), 4),
                      "studio_block_accuracy": round(float(accuracy_score(ys_te, m.predict(Xs_te))), 4)})
        print(f"  all-pool, weight={w_}: external {wrows[-1]['external_accuracy']:.3f} | studio {wrows[-1]['studio_block_accuracy']:.3f}", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "adaptation.json").write_text(json.dumps({
        "protocol": "calibration = first k files per letter of the external set; test = last 50% of that letter's files",
        "model": name, "sample_weight": WEIGHT, "n_test_images": int(len(yt)), "n_pool_images": int(len(pool)), "weight_sensitivity": wrows, "rows": rows}, indent=1), encoding="utf-8")
    print("saved results/adaptation.json")


if __name__ == "__main__":
    main()
