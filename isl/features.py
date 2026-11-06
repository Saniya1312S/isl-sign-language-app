"""
Feature engineering for hand landmarks.

Input  : `coords`  float array (N, 2, 21, 3)  - up to 2 hands, 21 landmarks, (x, y, z).
                   Hands are ordered by wrist x-position (image-left first).
         `n_hands` int array   (N,)           - 0, 1 or 2 hands actually detected.
Output : a 2-D feature matrix (N, n_features).

Feature "modes" (kept so we can run an honest ablation study):

    legacy_raw126  The ORIGINAL project's features: raw image coordinates, hand placed in the
                   first/second 63 slots according to MediaPipe's Left/Right label.
    sorted_raw     Same raw coordinates but hands ordered by position, not by label.
    norm           Translation- and scale-invariant: coordinates relative to the wrist,
                   divided by hand size, + number of hands + inter-hand geometry.
    norm_eng       `norm` + fingertip distances, pairwise fingertip distances, finger-curl angles.

Why normalise?  Raw coordinates change when the person moves closer/further from the camera
or shifts to the side, although the sign did not change. Normalised coordinates describe only
the SHAPE of the hand.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

FEATURE_MODES = ("legacy_raw126", "sorted_raw", "norm", "norm_eng")

WRIST, MIDDLE_MCP = 0, 9
TIPS = (4, 8, 12, 16, 20)                         # thumb, index, middle, ring, pinky tips
# (base, middle, tip) landmark triples used for the finger-curl angle of each finger
CURL = ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
EPS = 1e-6


# --------------------------------------------------------------------------- ordering helpers
def canonical_order(coords: np.ndarray, n_hands: np.ndarray) -> np.ndarray:
    """Return coords with the present hands sorted by wrist x (leftmost first)."""
    out = np.zeros_like(coords)
    for i in range(len(coords)):
        n = int(n_hands[i])
        if n == 0:
            continue
        order = np.argsort(coords[i, :n, WRIST, 0])
        out[i, :n] = coords[i, :n][order]
    return out


def mirror(coords: np.ndarray, n_hands: np.ndarray) -> np.ndarray:
    """Mirror the signer left<->right (x -> 1 - x) and restore the canonical hand order."""
    flipped = coords.copy()
    for i in range(len(flipped)):
        n = int(n_hands[i])
        flipped[i, :n, :, 0] = 1.0 - flipped[i, :n, :, 0]
    return canonical_order(flipped, n_hands)


# --------------------------------------------------------------------------- feature blocks
def _hand_scale(coords: np.ndarray) -> np.ndarray:
    """Hand size = 2-D distance wrist -> middle-finger base. Shape (N, 2)."""
    d = coords[:, :, MIDDLE_MCP, :2] - coords[:, :, WRIST, :2]
    return np.linalg.norm(d, axis=-1)


def _normalised(coords: np.ndarray, present: np.ndarray):
    """Wrist-centred, size-normalised coordinates (N, 2, 21, 3); zeros where no hand."""
    scale = _hand_scale(coords)
    scale = np.where(scale < EPS, 1.0, scale)
    rel = coords - coords[:, :, WRIST:WRIST + 1, :]
    rel = rel / scale[:, :, None, None]
    rel = rel * present[:, :, None, None]
    return rel, scale


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at point b formed by a-b-c, in radians (0..pi)."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + EPS)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _engineered(rel: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Per-hand engineered features (N, 2, 20): 5 tip distances, 10 tip-pair distances, 5 curls."""
    tips = rel[:, :, TIPS, :]                                         # (N, 2, 5, 3)
    tip_wrist = np.linalg.norm(tips, axis=-1)                         # (N, 2, 5)
    pairs = [np.linalg.norm(tips[:, :, i] - tips[:, :, j], axis=-1)
             for i in range(5) for j in range(i + 1, 5)]              # 10 x (N, 2)
    pairs = np.stack(pairs, axis=-1)                                  # (N, 2, 10)
    curls = np.stack([_angle(rel[:, :, a], rel[:, :, b], rel[:, :, c]) for a, b, c in CURL], axis=-1)
    feats = np.concatenate([tip_wrist, pairs, curls], axis=-1)
    return feats * present[:, :, None]


def _inter_hand(coords: np.ndarray, n_hands: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Geometry between the two hands (N, 4): wrist-to-wrist dx, dy, dz and distance, in hand sizes."""
    two = (n_hands == 2)
    mean_scale = np.where(two, scale.mean(axis=1), 1.0)
    mean_scale = np.where(mean_scale < EPS, 1.0, mean_scale)
    d = coords[:, 1, WRIST, :] - coords[:, 0, WRIST, :]
    d = d / mean_scale[:, None]
    dist = np.linalg.norm(d[:, :2], axis=-1, keepdims=True)
    out = np.concatenate([d, dist], axis=1)
    return out * two[:, None]


# --------------------------------------------------------------------------- public API
def build_features(coords: np.ndarray, n_hands: np.ndarray, mode: str = "norm_eng",
                   labels: np.ndarray | None = None) -> np.ndarray:
    """Vectorised feature builder.  `labels` ((N, 2) of 'Left'/'Right'/'') is only needed for legacy."""
    coords = np.asarray(coords, dtype=np.float32)
    n_hands = np.asarray(n_hands, dtype=int)
    n = len(coords)
    present = (np.arange(2)[None, :] < n_hands[:, None]).astype(np.float32)   # (N, 2)

    if mode == "sorted_raw":
        return coords.reshape(n, -1)

    if mode == "legacy_raw126":
        if labels is None:
            raise ValueError("legacy_raw126 needs the Left/Right labels")
        out = np.zeros((n, 126), dtype=np.float32)
        for i in range(n):
            for s in range(int(n_hands[i])):
                flat = coords[i, s].reshape(-1)
                if labels[i][s] == "Left":
                    out[i, :63] = flat            # original code: Left  -> first 63 values
                else:
                    out[i, 63:] = flat            # original code: Right -> last 63 values
        return out

    rel, scale = _normalised(coords, present)
    base = [rel.reshape(n, -1), n_hands[:, None].astype(np.float32),
            _inter_hand(coords, n_hands, scale)]
    if mode == "norm":
        return np.concatenate(base, axis=1).astype(np.float32)
    if mode == "norm_eng":
        eng = _engineered(rel, present).reshape(n, -1)
        return np.concatenate(base + [eng], axis=1).astype(np.float32)
    raise ValueError(f"unknown feature mode: {mode} (choose from {FEATURE_MODES})")


def feature_names(mode: str) -> List[str]:
    """Human-readable names, in the same order as build_features()."""
    axes = "xyz"
    if mode in ("sorted_raw", "legacy_raw126"):
        return [f"h{h}_lm{j}_{a}" for h in range(2) for j in range(21) for a in axes]
    names = [f"h{h}_lm{j}_{a}" for h in range(2) for j in range(21) for a in axes]
    names += ["n_hands", "wrist_dx", "wrist_dy", "wrist_dz", "wrist_dist"]
    if mode == "norm":
        return names
    tip = ["thumb", "index", "middle", "ring", "pinky"]
    for h in range(2):
        names += [f"h{h}_{t}_tip_to_wrist" for t in tip]
        names += [f"h{h}_{tip[i]}_{tip[j]}_tip_dist" for i in range(5) for j in range(i + 1, 5)]
        names += [f"h{h}_{t}_curl" for t in tip]
    return names


def hands_to_arrays(hands: Sequence) -> tuple:
    """Convert detected `Hand` objects to (coords(1,2,21,3), n_hands(1,), labels(1,2))."""
    hands = sorted(hands, key=lambda h: h.wrist_x)[:2]
    coords = np.zeros((1, 2, 21, 3), dtype=np.float32)
    labels = np.full((1, 2), "", dtype=object)
    for i, h in enumerate(hands):
        coords[0, i] = h.landmarks
        labels[0, i] = h.label
    return coords, np.array([len(hands)]), labels
