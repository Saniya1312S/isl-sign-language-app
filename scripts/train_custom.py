"""
Train an extra recogniser (WORDS or HINDI letters) from your own folder of images.

Folder layout (one sub-folder per class; the folder name becomes the label shown in the app):

    my_words/
        doctor/  img001.jpg  img002.jpg ...
        small/   ...
        water/   ...

Usage:
    python scripts/train_custom.py --name words --folder "C:/data/my_words"
    python scripts/train_custom.py --name hindi --folder "C:/data/hindi_letters"

It extracts landmarks (same MediaPipe settings as the alphabet model), builds the same normalised
features (+ mirror augmentation) and saves   models/<name>_model.joblib   +   models/<name>_meta.json.
The app detects those files at start-up and adds the mode to the "Recognition mode" menu.

HONEST EVALUATION WARNING
    If your images are frames from videos, a random split leaks near-identical frames into the test set and
    the accuracy will be too optimistic (see Dashboard > Evaluation honesty). This script therefore also
    reports a stricter "split by file order" score (first 80% of each class for training, last 20% for test),
    which keeps neighbouring frames together. Trust that number more.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import MODELS_DIR, RANDOM_STATE, RESULTS_DIR                       # noqa: E402
from isl.data import to_arrays                                                      # noqa: E402
from isl.features import build_features, mirror                                     # noqa: E402
from isl.hands import HandDetector                                                  # noqa: E402
from isl.model_io import save_bundle                                                # noqa: E402
from extract_landmarks import COORD_COLS, IMG_EXT, META_COLS, frame_index, row_for_image   # noqa: E402

warnings.filterwarnings("ignore")


def pipeline():
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", ExtraTreesClassifier(n_estimators=250, n_jobs=-1, random_state=RANDOM_STATE))])


def feats(df, augment):
    a = to_arrays(df)
    X = build_features(a["coords"], a["n_hands"], "norm_eng", a["labels"])
    y = a["y"]
    if augment:
        X = np.vstack([X, build_features(mirror(a["coords"], a["n_hands"]), a["n_hands"], "norm_eng", a["labels"])])
        y = np.concatenate([y, y])
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, choices=["words", "hindi"])
    ap.add_argument("--folder", required=True)
    ap.add_argument("--cap", type=int, default=300, help="max images per class (0 = all)")
    ap.add_argument("--min-images", type=int, default=20, help="skip classes with fewer usable images")
    args = ap.parse_args()

    det = HandDetector(static_image_mode=True)
    rows, t0 = [], time.time()
    classes = sorted(p for p in Path(args.folder).iterdir() if p.is_dir())
    if not classes:
        sys.exit(f"No class sub-folders found in {args.folder}")
    for cdir in classes:
        files = sorted((p for p in cdir.iterdir() if p.suffix.lower() in IMG_EXT), key=lambda p: (frame_index(p.name), p.name))
        if args.cap:
            files = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, min(args.cap, len(files))).round().astype(int))] if files else []
        got = [r for r in (row_for_image(p, det, args.name, cdir.name) for p in files) if r and r["n_hands"] > 0]
        print(f"  {cdir.name:25s} {len(got):4d}/{len(files):4d} images with a hand", flush=True)
        if len(got) >= args.min_images:
            rows.extend(got)
    df = pd.DataFrame(rows, columns=META_COLS + COORD_COLS)
    n_classes = df["label"].nunique()
    print(f"\n{len(df)} usable images, {n_classes} classes  ({time.time() - t0:.0f}s)")
    if n_classes < 2:
        sys.exit("Need at least 2 classes with enough images.")

    # (1) random split (optimistic if frames come from videos)
    tr, te = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"])
    m = pipeline().fit(*feats(tr, True))
    Xte, yte = feats(te, False)
    rand_acc = accuracy_score(yte, m.predict(Xte))
    # (2) stricter: last 20% of every class (file order) is the test set
    tr_parts, te_parts = [], []
    for _, g in df.groupby("label"):
        g = g.sort_values("frame_idx")
        k = int(len(g) * 0.8)
        tr_parts.append(g.iloc[:k])
        te_parts.append(g.iloc[k:])
    tr2, te2 = pd.concat(tr_parts), pd.concat(te_parts)
    m2 = pipeline().fit(*feats(tr2, True))
    Xte2, yte2 = feats(te2, False)
    strict_acc = accuracy_score(yte2, m2.predict(Xte2))
    strict_f1 = f1_score(yte2, m2.predict(Xte2), average="macro")
    print(f"random split accuracy : {rand_acc:.1%}   (optimistic)")
    print(f"strict (file order)   : {strict_acc:.1%}   macro-F1 {strict_f1:.3f}   <- trust this one")

    final = pipeline().fit(*feats(df, True))
    meta = {"name": args.name, "model_type": "Extra Trees", "feature_mode": "norm_eng", "mirror_augmentation": True,
            "n_training_images": int(len(df)), "n_classes": int(n_classes),
            "training_source": f"user folder: {Path(args.folder).name}",
            "evaluation": {"random_split_accuracy": round(rand_acc, 4), "strict_file_order_accuracy": round(strict_acc, 4),
                           "strict_macro_f1": round(strict_f1, 4)}}
    save_bundle(final, meta, MODELS_DIR / f"{args.name}_model.joblib", MODELS_DIR / f"{args.name}_meta.json")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"custom_{args.name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved models/{args.name}_model.joblib - restart the app to use it.")


if __name__ == "__main__":
    main()
