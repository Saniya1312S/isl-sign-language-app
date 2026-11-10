"""
STEP 2 - Exploratory data analysis (EDA) on the landmark table.

Writes results/eda.json (+ results/pca_sample.csv). The Streamlit dashboard only reads these small
files, so it opens instantly and every number on it can be traced back to this script.

Questions this script answers (each is a data-analyst style finding):
  1. How big is the data?  images / classes / per-class balance / per-source detection rate
  2. Is the data clean?    how many images have no hand found, 1 hand vs 2 hands per letter
  3. Is a random split safe?  -> near-duplicate test: how close is a "test" image to its nearest
     "training" image under a random split vs. under a block split?
  4. Is MediaPipe's Left/Right label a stable feature?  (it is not - see `handedness`)
  5. Do the two data sources look alike?  -> PCA of normalised features, coloured by source
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.config import LANDMARKS_CSV, RANDOM_STATE, RESULTS_DIR     # noqa: E402
from isl.data import blocks, load_landmarks, to_arrays              # noqa: E402
from isl.features import build_features                              # noqa: E402

BLOCK = 200


def overview(df: pd.DataFrame) -> dict:
    out = {"total_images": int(len(df)), "sources": {}}
    for src, g in df.groupby("source"):
        det = g[g.n_hands > 0]
        out["sources"][src] = {
            "images": int(len(g)),
            "classes": sorted(g["label"].unique().tolist()),
            "n_classes": int(g["label"].nunique()),
            "images_with_hand": int(len(det)),
            "detection_rate": round(len(det) / len(g), 4),
            "per_class": {k: int(v) for k, v in g.groupby("label").size().items()},
            "detection_by_class": {k: round(float((v.n_hands > 0).mean()), 4) for k, v in g.groupby("label")},
            "image_size": f"{int(g.img_w.mode()[0])}x{int(g.img_h.mode()[0])}",
            "hands_dist": {str(k): int(v) for k, v in g.n_hands.value_counts().sort_index().items()},
            "hands_by_class": {k: {str(h): int(c) for h, c in v.n_hands.value_counts().items()}
                               for k, v in g.groupby("label")},
        }
    return out


def handedness(df: pd.DataFrame) -> dict:
    """For one-hand images: how often does MediaPipe say 'Left' vs 'Right' for the SAME letter?"""
    one = df[df.n_hands == 1]
    res = {}
    for src, g in one.groupby("source"):
        tab = g.groupby("label")["h0_label"].apply(lambda s: float((s == "Left").mean()))
        res[src] = {"left_share_by_class": {k: round(v, 3) for k, v in tab.items()},
                    "classes_split_both_ways": int(((tab > 0.1) & (tab < 0.9)).sum()),
                    "n_classes": int(len(tab))}
    return res


def near_duplicate_test(df: pd.DataFrame, per_class: int = 150) -> dict:
    """Distance from a sample to its nearest neighbour of the same letter, under two split rules."""
    rng = np.random.RandomState(RANDOM_STATE)
    g = df[(df.source == "Indian") & (df.n_hands > 0) & (df.frame_idx >= 0)]
    rand_d, block_d, chain_d = [], [], []
    for label, cls in g.groupby("label"):
        if len(cls) < 400:
            continue
        a = to_arrays(cls)
        X = StandardScaler().fit_transform(build_features(a["coords"], a["n_hands"], "norm"))
        blk = (cls["frame_idx"].to_numpy() // BLOCK)
        idx = rng.choice(len(cls), size=min(per_class, len(cls)), replace=False)
        nn = NearestNeighbors(n_neighbors=len(cls)).fit(X) if len(cls) <= 1500 else None
        dist, ind = nn.kneighbors(X[idx])
        for r, i in enumerate(idx):
            others = ind[r][1:]                                  # skip itself
            rand_d.append(dist[r][1])                            # nearest neighbour anywhere in class
            far = others[blk[others] != blk[i]]                  # nearest in a different block
            if len(far):
                block_d.append(dist[r][list(ind[r]).index(far[0])])
            near_frame = others[np.abs(cls["frame_idx"].to_numpy()[others] - cls["frame_idx"].to_numpy()[i]) <= 2]
            chain_d.append(1.0 if len(near_frame) else 0.0)
    rand_d, block_d = np.array(rand_d), np.array(block_d)
    return {
        "block_size_frames": BLOCK,
        "median_nn_distance_random_split": round(float(np.median(rand_d)), 3),
        "median_nn_distance_block_split": round(float(np.median(block_d)), 3),
        "ratio_block_over_random": round(float(np.median(block_d) / max(np.median(rand_d), 1e-9)), 2),
        "share_nn_is_within_2_frames": round(float(np.mean(chain_d)), 3),
        "interpretation": ("Under a random split a test frame almost always has a near-identical twin in the "
                           "training set (its neighbouring video frame). Holding out whole blocks of frames "
                           "makes the nearest training example much further away."),
    }


def mean_shapes(df: pd.DataFrame) -> dict:
    """Average normalised hand skeleton per letter (what the classifier effectively 'sees')."""
    shapes = {}
    d = df[(df.source == "Indian") & (df.n_hands > 0)]
    for label, cls in d.groupby("label"):
        a = to_arrays(cls)
        nh = int(cls.n_hands.mode()[0])
        rows = a["n_hands"] == nh
        c = a["coords"][rows]
        X = build_features(c, a["n_hands"][rows], "norm")[:, :126].reshape(-1, 2, 21, 3)
        mean = X.mean(axis=0)[:nh]
        # place two hands side by side using the mean inter-hand offset for a readable picture
        if nh == 2:
            gap = np.median(build_features(c, a["n_hands"][rows], "norm")[:, 127:129], axis=0)
            mean[1, :, 0] += gap[0]
            mean[1, :, 1] += gap[1]
        shapes[label] = {"n_hands": nh, "points": np.round(mean[:, :, :2], 3).tolist()}
    return shapes


def pca_sample(df: pd.DataFrame, n_per_group: int = 60) -> pd.DataFrame:
    d = df[df.n_hands > 0].copy()
    parts = [g.sample(min(len(g), n_per_group), random_state=RANDOM_STATE) for _, g in d.groupby(["source", "label"])]
    s = pd.concat(parts).reset_index(drop=True)
    a = to_arrays(s)
    X = StandardScaler().fit_transform(build_features(a["coords"], a["n_hands"], "norm_eng"))
    p = PCA(n_components=3, random_state=RANDOM_STATE).fit(X)
    Z = p.transform(X)
    out = pd.DataFrame({"pc1": Z[:, 0].round(3), "pc2": Z[:, 1].round(3), "pc3": Z[:, 2].round(3),
                        "label": s["label"], "source": s["source"], "n_hands": s["n_hands"]})
    out.attrs["explained"] = [round(float(v), 4) for v in p.explained_variance_ratio_]
    return out


def main():
    df = load_landmarks(LANDMARKS_CSV)
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"{len(df):,} rows loaded from {LANDMARKS_CSV}")
    res = overview(df)
    res["handedness"] = handedness(df)
    print("near-duplicate test ...")
    res["near_duplicates"] = near_duplicate_test(df)
    print("mean shapes ...")
    res["mean_shapes"] = mean_shapes(df)
    pca = pca_sample(df)
    res["pca_explained_variance"] = pca.attrs["explained"]
    pca.to_csv(RESULTS_DIR / "pca_sample.csv", index=False)
    (RESULTS_DIR / "eda.json").write_text(json.dumps(res, indent=1), encoding="utf-8")

    for s, v in res["sources"].items():
        print(f"{s}: {v['images']:,} images, {v['n_classes']} classes, hand found in {v['detection_rate']:.1%}")
    nd = res["near_duplicates"]
    print(f"near-duplicate test: median NN distance random={nd['median_nn_distance_random_split']} "
          f"vs block={nd['median_nn_distance_block_split']} (x{nd['ratio_block_over_random']})")
    print("saved results/eda.json and results/pca_sample.csv")


if __name__ == "__main__":
    main()
