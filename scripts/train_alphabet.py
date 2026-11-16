"""
STEP 3 - Train and honestly evaluate the alphabet (A-Z) classifier.

The evaluation protocol is the heart of this project. The original code used ONE random split of
near-duplicate video frames (reported ~97%). Here the same data is evaluated three ways:

  P1  random split         - what the original project did (optimistic: test frames have twins in train)
  P2  block split          - whole blocks of consecutive frames held out (realistic for new captures)
  P3  external dataset     - ISL_Dataset_2: other people, hands and backgrounds (realistic for new users)

Stages:
    protocols   P1 vs P2 vs P3 with one fixed model                  -> results/protocols.json
    ablation    feature modes x mirror augmentation                  -> results/ablation.json
    learning    accuracy vs number of training images                -> results/learning_curve.json
    compare     7 classifiers, group-CV model selection, test       -> results/model_comparison.json
    final       best model retrained on all data + saved bundle      -> models/alphabet_model.joblib
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import ALPHABET_META, ALPHABET_MODEL, MODELS_DIR, RANDOM_STATE, RESULTS_DIR  # noqa: E402
from isl.data import blocks, load_landmarks, to_arrays                                         # noqa: E402
from isl.features import FEATURE_MODES, build_features, feature_names, mirror                  # noqa: E402
from isl.model_io import save_bundle                                                           # noqa: E402

warnings.filterwarnings("ignore")
TIE_TOLERANCE = 0.005    # CV macro-F1 difference treated as a tie
EXTERNAL_REPEAT = 10     # production model: each external image is repeated this many times
MAX_MODEL_MB = 60
STUDY_CAP = 600          # images per class used in the experiments
BLOCK = 200


# ------------------------------------------------------------------------------ PyTorch DL Estimator
class PyTorchResidualClassifier(BaseEstimator, ClassifierMixin):
    """PyTorch Residual MLP wrapped as an sklearn-compatible classifier."""
    def __init__(self, hidden_dim: int = 256, num_blocks: int = 2, epochs: int = 25,
                 lr: float = 1e-3, batch_size: int = 128, random_state: int = RANDOM_STATE):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        import torch
        import torch.nn as nn
        from isl.dl_model import ISLLandmarkNet

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.classes_ = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y], dtype=np.int64)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_idx, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_ = ISLLandmarkNet(
            in_features=X.shape[1],
            num_classes=len(self.classes_),
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks
        )
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = self.model_(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        # Fixed epochs, final weights (no best-epoch selection)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        import torch
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model_(X_t)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]


# ------------------------------------------------------------------------------ data loading & split
def cap_per_class(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Keep at most `cap` rows per class, spread evenly over the frame order."""
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
        mc, ml = mirror(a["coords"], a["n_hands"], a["labels"])
        Xm = build_features(mc, a["n_hands"], mode, ml)
        X, y = np.vstack([X, Xm]), np.concatenate([y, y])
    return X, y


def block_split(df: pd.DataFrame, test_size=0.2):
    """Hold out whole blocks of consecutive frames, chosen separately inside EVERY letter,
    so each letter has test data."""
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


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over the classes that really occur in y_true."""
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
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE))])


def models():
    """All 7 classifiers evaluated through the same pipeline."""
    sc = lambda: StandardScaler()
    m = {
        "Logistic Regression": Pipeline([("scaler", sc()), ("clf", LogisticRegression(max_iter=300, C=1.0))]),
        "k-Nearest Neighbours": Pipeline([("scaler", sc()), ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
        "Random Forest": Pipeline([("scaler", sc()), ("clf", RandomForestClassifier(
            n_estimators=120, n_jobs=-1, random_state=RANDOM_STATE))]),
        "Extra Trees": Pipeline([("scaler", sc()), ("clf", ExtraTreesClassifier(
            n_estimators=150, n_jobs=-1, random_state=RANDOM_STATE))]),
        "Gradient Boosting (Hist)": Pipeline([("scaler", sc()), ("clf", HistGradientBoostingClassifier(
            max_iter=50, learning_rate=0.1, random_state=RANDOM_STATE))]),
        "Neural Network (MLP)": Pipeline([("scaler", sc()), ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128), early_stopping=False, max_iter=40, random_state=RANDOM_STATE))]),
    }
    # Add PyTorch DL if torch is available
    try:
        import torch  # noqa: F401
        m["PyTorch ResNet-MLP"] = Pipeline([("scaler", sc()), ("clf", PyTorchResidualClassifier(
            hidden_dim=256, num_blocks=2, epochs=25, lr=1e-3, random_state=RANDOM_STATE))])
    except ImportError:
        pass
    return m


# ------------------------------------------------------------------------------ stages
def stage_protocols(indian, ext, ext_all):
    print("== protocols: random vs block vs external", flush=True)
    study = cap_per_class(indian, STUDY_CAP).reset_index(drop=True)
    out = {"model": "Random Forest (100 trees)", "features": "norm_eng", "cap_per_class": STUDY_CAP,
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
    a_te = to_arrays(te)
    mc_te, ml_te = mirror(a_te["coords"], a_te["n_hands"], a_te["labels"])
    yte = a_te["y"]

    rows = []
    for mode in FEATURE_MODES:
        for aug in (False, True):
            t0 = time.time()
            Xtr, ytr = featurize(tr, mode, augment=aug)
            m = fast_model().fit(Xtr, ytr)

            # 1. Standard held-out blocks
            Xte = build_features(a_te["coords"], a_te["n_hands"], mode, a_te["labels"])
            s_block = score(m, Xte, yte)

            # 2. Mirrored held-out blocks
            Xte_m = build_features(mc_te, a_te["n_hands"], mode, ml_te)
            s_mirr = score(m, Xte_m, yte)

            # 3. External dataset
            ext_s = external_scores(m, mode, ext, ext_all)

            row = {
                "features": mode,
                "mirror_augmentation": aug,
                "n_features": int(Xtr.shape[1]),
                "block_test": s_block,
                "mirrored_block_test": s_mirr,
                "external": ext_s,
                "seconds": round(time.time() - t0, 1)
            }
            rows.append(row)
            print(f"  {mode:14s} aug={aug!s:5s} block={row['block_test']['accuracy']:.3f} "
                  f"mirr={row['mirrored_block_test']['accuracy']:.3f} ext={ext_s['accuracy']:.3f}", flush=True)
    save("ablation.json", {"model": "Random Forest (100 trees)", "rows": rows})


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
    print("== compare 7 classifiers (group 3-fold CV for selection)", flush=True)
    study = cap_per_class(indian, STUDY_CAP).reset_index(drop=True)
    tr, te = block_split(study)
    Xtr, ytr = featurize(tr, "norm_eng", augment=True)
    groups = np.concatenate([blocks(tr, BLOCK)] * 2)
    Xte, yte = featurize(te, "norm_eng")

    rows = []
    all_models = models()
    for name, pipe in all_models.items():
        t0 = time.time()
        cv = []
        for a, b in GroupKFold(n_splits=3).split(Xtr, ytr, groups):
            m = clone(pipe).fit(Xtr[a], ytr[a])
            cv.append(macro_f1(ytr[b], m.predict(Xtr[b])))
        m = clone(pipe).fit(Xtr, ytr)
        fit_s = time.time() - t0

        # Un-amortized single-sample latency measurement over 100 individual samples
        sample_times = []
        for row in Xte[:100]:
            t_s = time.perf_counter()
            m.predict(row[None, :])
            sample_times.append(time.perf_counter() - t_s)
        ms = float(np.median(sample_times) * 1000)

        # File size
        if name == "PyTorch ResNet-MLP":
            sz_buf = io.BytesIO()
            import torch
            torch.save(m.named_steps["clf"].model_.state_dict(), sz_buf)
            sz = round(sz_buf.getbuffer().nbytes / 1e6, 2)
        else:
            sz = size_mb(m)

        row = {"model": name, "cv_macro_f1": round(float(np.mean(cv)), 4), "cv_std": round(float(np.std(cv)), 4),
               "train_accuracy": score(m, Xtr, ytr)["accuracy"],
               "block_test": score(m, Xte, yte), "external": external_scores(m, "norm_eng", ext, ext_all),
               "fit_seconds": round(fit_s, 1), "predict_ms_per_sample": round(ms, 3),
               "model_size_mb": sz}
        row["generalisation_gap"] = round(row["train_accuracy"] - row["block_test"]["accuracy"], 4)
        rows.append(row)
        print(f"  {name:26s} cv={row['cv_macro_f1']:.3f} block={row['block_test']['accuracy']:.3f} "
              f"ext={row['external']['accuracy']:.3f} fit={fit_s:.0f}s", flush=True)

    # Selection rule declared BEFORE looking at test sets:
    # Among scikit-learn models (non-DL, must work without PyTorch) within 0.005 CV macro-F1 of the best, take the smallest file
    sklearn_rows = [r for r in rows if r["model"] != "PyTorch ResNet-MLP"]
    best_f1 = max(r["cv_macro_f1"] for r in sklearn_rows)
    tied = [r for r in sklearn_rows if r["cv_macro_f1"] >= best_f1 - TIE_TOLERANCE]
    chosen = min(tied, key=lambda r: r["model_size_mb"])["model"]

    save("model_comparison.json", {
        "features": "norm_eng + mirror augmentation",
        "cap_per_class": STUDY_CAP,
        "selected_by": (f"3-fold GroupKFold macro-F1 on training blocks; scikit-learn models within "
                        f"{TIE_TOLERANCE:.3f} of the best are tied and the smallest file wins"),
        "selected_model": chosen,
        "rows": rows
    })
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
    chosen = chosen or json.loads(cmp_path.read_text(encoding="utf-8"))["selected_model"]
    all_models = models()
    pipe = all_models[chosen]

    # (a) Evaluation copy: trained on block-train only -> honest per-class results, confusion, calibration
    study = cap_per_class(indian, STUDY_CAP).reset_index(drop=True)
    tr, te = block_split(study)
    eval_model = clone(pipe).fit(*featurize(tr, "norm_eng", augment=True))
    reports = {"model": chosen,
               "block_test": per_class_report(eval_model, "norm_eng", te, "Indian (held-out blocks)"),
               "external": per_class_report(eval_model, "norm_eng", ext, "ISL_Dataset_2 (external)")}

    # Feature importance computed from model
    clf = eval_model.named_steps["clf"]
    raw_imp = None
    if hasattr(clf, "feature_importances_"):                       # tree models
        raw_imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):                                    # linear models: mean |standardised coefficient|
        raw_imp = np.abs(clf.coef_).mean(axis=0)
        raw_imp = raw_imp / raw_imp.sum()
    if raw_imp is not None:
        names = feature_names("norm_eng")
        imp = pd.Series(raw_imp, index=names).sort_values(ascending=False)
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
        tot = sum(groups.values()) or 1.0
        reports["importance_by_group"] = {k: round(float(v / tot), 4) for k, v in groups.items()}
    save("final_reports.json", reports)

    # (b) Production model: trained on ALL Indian frames + mirror augmentation + external images
    prod, used_cap = None, 0
    for cap in (0, 800, 600, 400, 250):
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
    cmp_data = json.loads(cmp_path.read_text(encoding="utf-8"))
    chosen_row = next(r for r in cmp_data["rows"] if r["model"] == chosen)
    meta = {
        "name": "alphabet", "model_type": chosen, "feature_mode": "norm_eng", "mirror_augmentation": True,
        "n_training_images": int(indian_used), "n_training_rows_after_augmentation": int(len(y)),
        "images_per_class_cap": int(used_cap) or "none",
        "training_source": "Indian (A-Z) frames where MediaPipe found a hand + ISL_Dataset_2 images (repeated x%d, mirror-augmented)" % EXTERNAL_REPEAT,
        "n_external_images_in_training": int(len(ext)),
        "evaluation": {"block_split_accuracy": chosen_row["block_test"],
                       "external_accuracy": chosen_row["external"]},
        "note": ("Evaluation numbers come from a copy trained on block-train only and never shown the external data. The saved "
                 "production model is retrained on all Indian frames PLUS the external images, so the external accuracy of "
                 "the production model cannot be measured (it has seen those images) - the evaluation numbers are the honest ones."),
    }
    save_bundle(prod, meta, ALPHABET_MODEL, ALPHABET_META)
    print(f"  saved {ALPHABET_MODEL.name} ({ALPHABET_MODEL.stat().st_size / 1e6:.2f} MB)", flush=True)

    # (c) Also export the optional PyTorch DL model with primitive weights_only=True compatible format
    try:
        import torch
        from isl.dl_model import ISLLandmarkNet
        print("== exporting optional PyTorch DL model (.pth)", flush=True)
        dl_clf = PyTorchResidualClassifier(hidden_dim=256, num_blocks=2, epochs=25, lr=1e-3, random_state=RANDOM_STATE)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        dl_clf.fit(X_scaled, y)

        dl_checkpoint = {
            "model_state_dict": dl_clf.model_.state_dict(),
            "classes": [str(c) for c in dl_clf.classes_],
            "feature_mode": "norm_eng",
            "in_features": int(X.shape[1]),
            "hidden_dim": 256,
            "num_blocks": 2,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
        dl_path = MODELS_DIR / "isl_dl_model.pth"
        torch.save(dl_checkpoint, dl_path)
        print(f"  saved {dl_path.name} ({dl_path.stat().st_size / 1e6:.2f} MB)", flush=True)
    except Exception as e:
        print(f"  (PyTorch export skipped: {e})", flush=True)


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
