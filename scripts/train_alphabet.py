"""
STEP 3 - Train and honestly evaluate the alphabet (A-Z) classifier.

The evaluation protocol is the heart of this project. The original code used ONE random split of
near-duplicate video frames (reported ~97%). Here the same data is evaluated three ways:

  P1  random split          - what the original project did   (optimistic: test frames have twins in train)
  P2  block split          - whole blocks of consecutive frames held out (realistic for new captures)
  P3  external dataset     - ISL_Dataset_2: other people, hands and backgrounds (realistic for new users)

Stages (run them one by one or all together; each saves a JSON file in results/):
    protocols   P1 vs P2 vs P3 with one fixed model                  -> results/protocols.json
    ablation    feature modes x mirror augmentation                  -> results/ablation.json
    learning    accuracy vs number of training images                -> results/learning_curve.json
    compare     6 classifiers, group-CV model selection, test       -> results/model_comparison.json
    final       best model retrained on all data + saved bundle      -> models/alphabet_model.joblib

Usage:  python scripts/train_alphabet.py --stage all        (or --stage protocols, etc.)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import ALPHABET_META, ALPHABET_MODEL, LANDMARKS_CSV, RANDOM_STATE, RESULTS_DIR   # noqa: E402
from isl.data import blocks, load_landmarks, to_arrays                                           # noqa: E402
from isl.features import FEATURE_MODES, build_features, feature_names, mirror                    # noqa: E402
from isl.model_io import save_bundle                                                             # noqa: E402

warnings.filterwarnings("ignore")
TIE_TOLERANCE = 0.005    # CV macro-F1 difference treated as a tie
EXTERNAL_REPEAT = 10     # production model: each external image is repeated this many times (it is the only diverse data we have)
MAX_MODEL_MB = 60        # the saved model must stay small enough for GitHub / a free cloud server
STUDY_CAP = 600          # images per class used in the experiments (see `learning` stage for why)
BLOCK = 200


# ------------------------------------------------------------------------------ data
def cap_per_class(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Keep at most `cap` rows per class, spread evenly over the frame order (not just the first ones)."""
    if not cap:
        return df
    parts = []
    for _, g in df.groupby("label"):
        g = g.sort_values("frame_idx")
        idx = np.linspace(0, len(g) - 1, min(cap, len(g))).round().astype(int)
        parts.append(g.iloc[np.unique(idx)])
    return pd.concat(parts)


def load():
    df = load_landmarks()
    indian = df[(df.source == "Indian") & (df.n_hands > 0)].reset_index(drop=True)
    ext_all = df[df.source == "ISL_Dataset_2"].reset_index(drop=True)
    ext = ext_all[ext_all.n_hands > 0].reset_index(drop=True)
    return df, indian, ext, ext_all


def featurize(df: pd.DataFrame, mode: str, augment: bool = False):
    a = to_arrays(df)
    X = build_features(a["coords"], a["n_hands"], mode, a["labels"])
    y = a["y"]
    if augment:
        Xm = build_features(mirror(a["coords"], a["n_hands"]), a["n_hands"], mode, a["labels"])
        X, y = np.vstack([X, Xm]), np.concatenate([y, y])
    return X, y


def block_split(df: pd.DataFrame, test_size=0.2):
    """Hold out whole blocks of consecutive frames, chosen separately inside EVERY letter, so each letter has test data.
    (Picking blocks at random over all letters at once can leave some letters without any test block.)"""
    rng = np.random.RandomState(RANDOM_STATE)
    group = blocks(df, BLOCK)
    label = df["label"].to_numpy()
    test_blocks = set()
    for lab in np.unique(label):
        ids = np.unique(group[label == lab])
        n_test = max(1, int(round(test_size * len(ids))))
        test_blocks.update(rng.choice(ids, size=n_test, replace=False).tolist())
    mask = np.array([g in test_blocks for g in group])
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def random_split(df: pd.DataFrame, test_size=0.2):
    tr, te = next(StratifiedShuffleSplit(1, test_size=test_size, random_state=RANDOM_STATE).split(df, df["label"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def models():
    """Six classifiers. All except tree models get a StandardScaler inside the Pipeline."""
    sc = lambda: StandardScaler()
    return {
        "Logistic Regression": Pipeline([("scaler", sc()), ("clf", LogisticRegression(max_iter=300, C=1.0))]),
        "k-Nearest Neighbours": Pipeline([("scaler", sc()), ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
        "Random Forest": Pipeline([("scaler", sc()), ("clf", RandomForestClassifier(
            n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE))]),
        "Extra Trees": Pipeline([("scaler", sc()), ("clf", ExtraTreesClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))]),
        "Gradient Boosting (Hist)": Pipeline([("scaler", sc()), ("clf", HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.1, random_state=RANDOM_STATE))]),
        "Neural Network (MLP)": Pipeline([("scaler", sc()), ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=80, random_state=RANDOM_STATE))]),
    }


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over the classes that really occur in y_true.
    (A held-out block split may not contain every letter; a letter with no true examples must not count as F1 = 0.)"""
    return float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro", zero_division=0))


def score(model, X, y):
    p = model.predict(X)
    return {"accuracy": round(float(accuracy_score(y, p)), 4), "macro_f1": round(macro_f1(y, p), 4)}


def external_scores(model, mode, ext, ext_all):
    """Accuracy on the external set, restricted to detected hands and 'end-to-end' (misses = wrong)."""
    Xe, ye = featurize(ext, mode)
    classes = set(model.classes_)
    keep = np.array([c in classes for c in ye])
    s = score(model, Xe[keep], ye[keep])
    s["n_images_with_hand"] = int(keep.sum())
    s["end_to_end_accuracy"] = round(float((model.predict(Xe[keep]) == ye[keep]).sum() / len(ext_all)), 4)
    s["n_images_total"] = int(len(ext_all))
    return s


def save(name, obj):
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  saved results/{name}", flush=True)


def size_mb(model) -> float:
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return round(buf.getbuffer().nbytes / 1e6, 2)


def fast_model():
    return Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(
        n_estimators=120, n_jobs=-1, random_state=RANDOM_STATE))])


# ------------------------------------------------------------------------------ stages
def stage_protocols(indian, ext, ext_all):
    print("== protocols: random vs block vs external", flush=True)
    study = cap_per_class(indian, STUDY_CAP).reset_index(drop=True)
    out = {"model": "Random Forest (120 trees)", "features": "norm_eng", "cap_per_class": STUDY_CAP,
           "block_size_frames": BLOCK, "n_images_used": int(len(study))}
    for name, splitter in (("P1_random_split", random_split), ("P2_block_split", block_split)):
        tr, te = splitter(study)
        Xtr, ytr = featurize(tr, "norm_eng")
        m = fast_model().fit(Xtr, ytr)
        Xte, yte = featurize(te, "norm_eng")
        out[name] = {**score(m, Xte, yte), "n_train": int(len(tr)), "n_test": int(len(te))}
        print(f"  {name}: {out[name]}", flush=True)
    tr, _ = block_split(study)
    m = fast_model().fit(*featurize(tr, "norm_eng"))
    out["P3_external"] = external_scores(m, "norm_eng", ext, ext_all)
    print(f"  P3_external: {out['P3_external']}", flush=True)
    save("protocols.json", out)


def stage_ablation(indian, ext, ext_all):
    print("== ablation: feature modes x mirror augmentation", flush=True)
    study = cap_per_class(indian, STUDY_CAP).reset_index(drop=True)
    tr, te = block_split(study)
    rows = []
    for mode in FEATURE_MODES:
        for aug in (False, True):
            t0 = time.time()
            Xtr, ytr = featurize(tr, mode, augment=aug)
            m = fast_model().fit(Xtr, ytr)
            Xte, yte = featurize(te, mode)
            ext_s = external_scores(m, mode, ext, ext_all)
            row = {"features": mode, "mirror_augmentation": aug, "n_features": int(Xtr.shape[1]),
                   "block_test": score(m, Xte, yte), "external": ext_s,
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"  {mode:14s} aug={aug!s:5s} block={row['block_test']['accuracy']:.3f} "
                  f"external={ext_s['accuracy']:.3f}", flush=True)
    save("ablation.json", {"model": "Random Forest (120 trees)", "rows": rows})


def stage_learning(indian, ext, ext_all):
    print("== learning curve", flush=True)
    full_tr, te = block_split(indian)
    rows = []
    for n in (10, 25, 50, 100, 200, 400, 800):
        sub = cap_per_class(full_tr, n)
        m = fast_model().fit(*featurize(sub, "norm_eng", augment=True))
        Xte, yte = featurize(te, "norm_eng")
        rows.append({"images_per_class": n, "n_train": int(len(sub)),
                     "block_test_accuracy": score(m, Xte, yte)["accuracy"],
                     "external_accuracy": external_scores(m, "norm_eng", ext, ext_all)["accuracy"]})
        print(f"  {n:4d}/class: block={rows[-1]['block_test_accuracy']:.3f} external={rows[-1]['external_accuracy']:.3f}", flush=True)
    save("learning_curve.json", {"rows": rows})


def stage_compare(indian, ext, ext_all):
    print("== compare 6 classifiers (group 3-fold CV for selection)", flush=True)
    study = cap_per_class(indian, STUDY_CAP).reset_index(drop=True)
    tr, te = block_split(study)
    Xtr, ytr = featurize(tr, "norm_eng", augment=True)
    groups = np.concatenate([blocks(tr, BLOCK)] * 2)
    Xte, yte = featurize(te, "norm_eng")
    rows, best = [], None
    for name, pipe in models().items():
        t0 = time.time()
        cv = []
        for a, b in GroupKFold(n_splits=3).split(Xtr, ytr, groups):
            m = clone(pipe).fit(Xtr[a], ytr[a])
            cv.append(macro_f1(ytr[b], m.predict(Xtr[b])))
        m = clone(pipe).fit(Xtr, ytr)
        fit_s = time.time() - t0
        t1 = time.time(); m.predict(Xte[:500]); ms = (time.time() - t1) / 500 * 1000
        row = {"model": name, "cv_macro_f1": round(float(np.mean(cv)), 4), "cv_std": round(float(np.std(cv)), 4),
               "train_accuracy": score(m, Xtr, ytr)["accuracy"],
               "block_test": score(m, Xte, yte), "external": external_scores(m, "norm_eng", ext, ext_all),
               "fit_seconds": round(fit_s, 1), "predict_ms_per_sample": round(ms, 3),
               "model_size_mb": size_mb(m)}
        row["generalisation_gap"] = round(row["train_accuracy"] - row["block_test"]["accuracy"], 4)
        rows.append(row)
        print(f"  {name:26s} cv={row['cv_macro_f1']:.3f} block={row['block_test']['accuracy']:.3f} "
              f"ext={row['external']['accuracy']:.3f} fit={fit_s:.0f}s", flush=True)
    best_f1 = max(r["cv_macro_f1"] for r in rows)
    tied = [r for r in rows if r["cv_macro_f1"] >= best_f1 - TIE_TOLERANCE]
    chosen = min(tied, key=lambda r: r["model_size_mb"])["model"]     # among near-ties prefer the smallest model
    save("model_comparison.json", {"features": "norm_eng + mirror augmentation", "cap_per_class": STUDY_CAP,
                                   "selected_by": (f"3-fold GroupKFold macro-F1 on training blocks; models within "
                                                   f"{TIE_TOLERANCE:.3f} of the best are tied and the smallest file wins"),
                                   "selected_model": chosen, "rows": rows})
    return chosen


def per_class_report(model, mode, df, name):
    X, y = featurize(df, mode)
    keep = np.array([c in set(model.classes_) for c in y])
    X, y = X[keep], y[keep]
    p = model.predict(X)
    labels = sorted(set(y) | set(p))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    cm = confusion_matrix(y, p, labels=labels)
    pairs = [{"true": labels[i], "pred": labels[j], "count": int(cm[i, j])}
             for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda r: -r["count"])
    # confidence / coverage trade-off
    proba = model.predict_proba(X)
    conf, top = proba.max(axis=1), model.classes_[proba.argmax(axis=1)]
    curve = []
    for t in np.arange(0.0, 0.96, 0.05):
        sel = conf >= t
        curve.append({"threshold": round(float(t), 2), "coverage": round(float(sel.mean()), 4),
                      "accuracy": round(float((top[sel] == y[sel]).mean()), 4) if sel.any() else None})
    return {"dataset": name, "labels": labels, "confusion_matrix": cm.tolist(),
            "per_class": [{"label": l, "precision": round(float(a), 3), "recall": round(float(b), 3),
                           "f1": round(float(c), 3), "support": int(d)}
                          for l, a, b, c, d in zip(labels, pr, rc, f1, sup) if d > 0],
            "top_confusions": pairs[:12], "confidence_curve": curve}


def stage_final(indian, ext, ext_all, chosen=None):
    print("== final model", flush=True)
    cmp_path = RESULTS_DIR / "model_comparison.json"
    chosen = chosen or json.loads(cmp_path.read_text())["selected_model"]
    pipe = models()[chosen]

    # (a) evaluation copy: trained on block-train only -> honest per-class results, confusion, calibration
    study = cap_per_class(indian, STUDY_CAP).reset_index(drop=True)
    tr, te = block_split(study)
    eval_model = clone(pipe).fit(*featurize(tr, "norm_eng", augment=True))
    reports = {"model": chosen,
               "block_test": per_class_report(eval_model, "norm_eng", te, "Indian (held-out blocks)"),
               "external": per_class_report(eval_model, "norm_eng", ext, "ISL_Dataset_2 (external)")}
    # permutation-free importance if available
    clf = eval_model.named_steps["clf"]
    raw_imp = None
    if hasattr(clf, "feature_importances_"):                       # tree models
        raw_imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):                                    # linear models: mean |standardised coefficient|
        raw_imp = np.abs(clf.coef_).mean(axis=0)
        raw_imp = raw_imp / raw_imp.sum()
    if raw_imp is not None:
        imp = pd.Series(raw_imp, index=feature_names("norm_eng")).sort_values(ascending=False)
        reports["feature_importance"] = [{"feature": k, "importance": round(float(v), 5)} for k, v in imp.head(25).items()]
        groups = {"hand shape (21 landmarks)": 0.0, "inter-hand geometry": 0.0, "fingertip distances": 0.0,
                  "finger curl angles": 0.0, "number of hands": 0.0}
        for k, v in imp.items():
            if k == "n_hands":
                groups["number of hands"] += v
            elif k.startswith("wrist_"):
                groups["inter-hand geometry"] += v
            elif "_tip_dist" in k or "_tip_to_wrist" in k:
                groups["fingertip distances"] += v
            elif k.endswith("_curl"):
                groups["finger curl angles"] += v
            else:
                groups["hand shape (21 landmarks)"] += v
        reports["importance_by_group"] = {k: round(float(v), 4) for k, v in groups.items()}
    save("final_reports.json", reports)

    # (b) production model: trained on ALL Indian frames + mirror augmentation
    prod, used_cap = None, 0
    for cap in (0, 800, 600, 400, 250):                      # 0 = every frame; shrink only if the file is too big
        sub = cap_per_class(indian, cap)
        X, y = featurize(sub, "norm_eng", augment=True)
        if EXTERNAL_REPEAT and len(ext):
            Xe, ye = featurize(ext, "norm_eng", augment=True)
            X = np.vstack([X] + [Xe] * EXTERNAL_REPEAT)
            y = np.concatenate([y] + [ye] * EXTERNAL_REPEAT)
        prod = clone(pipe).fit(X, y)
        used_cap = cap
        if size_mb(prod) <= MAX_MODEL_MB:
            break
        print(f"  model is {size_mb(prod):.0f} MB with cap={cap or 'all'} -> retrying with fewer images", flush=True)
    indian_used = len(sub)
    meta = {
        "name": "alphabet", "model_type": chosen, "feature_mode": "norm_eng", "mirror_augmentation": True,
        "n_training_images": int(indian_used), "n_training_rows_after_augmentation": int(len(y)),
        "images_per_class_cap": int(used_cap) or "none",
        "training_source": "Indian (A-Z) frames where MediaPipe found a hand + ISL_Dataset_2 images (repeated x%d, mirror-augmented)" % EXTERNAL_REPEAT,
        "n_external_images_in_training": int(len(ext)),
        "evaluation": {"block_split_accuracy": reports["block_test"] and next(
                           r for r in json.loads(cmp_path.read_text())["rows"] if r["model"] == chosen)["block_test"],
                       "external_accuracy": next(
                           r for r in json.loads(cmp_path.read_text())["rows"] if r["model"] == chosen)["external"]},
        "note": ("Evaluation numbers come from a copy trained on block-train only and never shown the external data. The saved "
                 "production model is retrained on all Indian frames PLUS the external images, so the external accuracy of "
                 "the production model cannot be measured (it has seen those images) - the evaluation numbers are the honest ones."),
    }
    save_bundle(prod, meta, ALPHABET_MODEL, ALPHABET_META)
    print(f"  saved {ALPHABET_MODEL.name} ({ALPHABET_MODEL.stat().st_size / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "protocols", "ablation", "learning", "compare", "final"])
    args = ap.parse_args()
    _, indian, ext, ext_all = load()
    print(f"Indian: {len(indian):,} images with a hand | external: {len(ext):,} with a hand of {len(ext_all):,}", flush=True)
    s = args.stage
    if s in ("all", "protocols"): stage_protocols(indian, ext, ext_all)
    if s in ("all", "ablation"): stage_ablation(indian, ext, ext_all)
    if s in ("all", "learning"): stage_learning(indian, ext, ext_all)
    chosen = stage_compare(indian, ext, ext_all) if s in ("all", "compare") else None
    if s in ("all", "final"): stage_final(indian, ext, ext_all, chosen)


if __name__ == "__main__":
    main()
