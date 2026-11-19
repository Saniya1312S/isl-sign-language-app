"""
Feature engineering for Indian Sign Language hand landmarks.

Input  : `coords`  float array (N, 2, 21, 3) - up to 2 hands, 21 landmarks, (x, y, z).
         `n_hands` int array   (N,)          - 0, 1 or 2 hands actually detected.
Output : a 2-D feature matrix (N, n_features).

Feature modes:
    robust_isl    [RECOMMENDED / SOTA] Rotation-invariant, scale-invariant, translation-invariant,
                  handedness-normalized, with joint angles, fingertip distances, extension ratios,
                  and 2-hand contact matrix (226 features).
    norm_eng      Wrist-centred, scale-normalized coords + fingertip distances + curls (171 features).
    norm          Wrist-centred, scale-normalized coords + inter-hand geometry (131 features).
    sorted_raw    Raw coordinates ordered by wrist-x position (126 features).
    legacy_raw126 Raw coordinates placed by MediaPipe Left/Right label (126 features).
"""
from __future__ import annotations

from typing import List, Sequence
import numpy as np

FEATURE_MODES = ("robust_isl", "norm_eng", "norm", "sorted_raw", "legacy_raw126")

WRIST, MIDDLE_MCP = 0, 9
TIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky tips
CURL = ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))

# 15 joint angle triples: (base, joint, tip)
JOINT_ANGLES = (
    # Thumb: CMC, MCP, IP
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    # Index: MCP, PIP, DIP
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    # Middle: MCP, PIP, DIP
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    # Ring: MCP, PIP, DIP
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    # Pinky: MCP, PIP, DIP
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
)

EPS = 1e-6


# --------------------------------------------------------------------------- ordering helpers
def canonical_order(coords: np.ndarray, n_hands: np.ndarray) -> np.ndarray:
    """Return coords with present hands sorted by wrist x (image-left first)."""
    out = np.zeros_like(coords)
    for i in range(len(coords)):
        n = int(n_hands[i])
        if n == 0:
            continue
        order = np.argsort(coords[i, :n, WRIST, 0])
        out[i, :n] = coords[i, :n][order]
    return out


def mirror(coords: np.ndarray, n_hands: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Mirror the signer left<->right (x -> 1 - x), swap Left<->Right labels and restore canonical hand order."""
    flipped = coords.copy()
    for i in range(len(flipped)):
        n = int(n_hands[i])
        flipped[i, :n, :, 0] = 1.0 - flipped[i, :n, :, 0]
    out_coords = canonical_order(flipped, n_hands)
    if labels is None:
        return out_coords

    out_labels = np.zeros_like(labels, dtype=object)
    for i in range(len(labels)):
        n = int(n_hands[i])
        if n == 0:
            continue
        order = np.argsort(flipped[i, :n, WRIST, 0])
        swapped = [("Right" if l == "Left" else ("Left" if l == "Right" else str(l))) for l in labels[i, :n]]
        for j, idx in enumerate(order):
            out_labels[i, j] = swapped[idx]
    dtype_out = labels.dtype if labels.dtype.itemsize >= 20 else "<U10"
    return out_coords, out_labels.astype(dtype_out)


def _compute_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at joint b formed by vectors (a - b) and (c - b). Shape (N, ...)."""
    v1 = a - b
    v2 = c - b
    v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True) + EPS
    v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True) + EPS
    cos = np.sum((v1 / v1_norm) * (v2 / v2_norm), axis=-1)
    return np.arccos(np.clip(cos, -1.0, 1.0))


# --------------------------------------------------------------------------- Robust SOTA Features
def _extract_robust_features(coords: np.ndarray, n_hands: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """
    State-of-the-art rotation-invariant, scale-invariant, handedness-normalized geometric features.
    Total: 226 dimensions.
    """
    N = len(coords)
    present = (np.arange(2)[None, :] < n_hands[:, None]).astype(np.float32)  # (N, 2)

    # 1. Wrist subtraction (translation invariance)
    rel = coords - coords[:, :, WRIST:WRIST + 1, :]

    # 2. Scale normalization: distance wrist -> middle MCP
    scale = np.linalg.norm(rel[:, :, MIDDLE_MCP, :2], axis=-1)  # (N, 2)
    scale = np.where(scale < EPS, 1.0, scale)

    # 3. In-plane rotation alignment: rotate so wrist -> middle MCP points along unit vector (0, -1)
    v = rel[:, :, MIDDLE_MCP, :2]  # (N, 2, 2)
    current_angle = np.arctan2(v[:, :, 1], v[:, :, 0])  # (N, 2)
    target_angle = -np.pi / 2.0
    rot_angle = target_angle - current_angle  # (N, 2)

    cos_a = np.cos(rot_angle)[:, :, None]  # (N, 2, 1)
    sin_a = np.sin(rot_angle)[:, :, None]  # (N, 2, 1)

    rx = rel[:, :, :, 0] * cos_a - rel[:, :, :, 1] * sin_a
    ry = rel[:, :, :, 0] * sin_a + rel[:, :, :, 1] * cos_a
    rz = rel[:, :, :, 2]

    norm_coords = np.stack([rx / scale[:, :, None], ry / scale[:, :, None], rz / scale[:, :, None]], axis=-1)

    # 4. Handedness normalization for single-hand signs:
    # If exactly 1 hand and it's labeled 'Left', mirror x -> -x so left and right hands share the same coordinate frame
    if labels is not None:
        single_left = (n_hands == 1) & (labels[:, 0] == 'Left')
        norm_coords[single_left, 0, :, 0] = -norm_coords[single_left, 0, :, 0]

    norm_coords = norm_coords * present[:, :, None, None]
    flat_lms = norm_coords.reshape(N, -1)  # 126

    # 5. Joint angles: 15 per hand = 30
    angles_list = []
    for h in range(2):
        for (a, b, c) in JOINT_ANGLES:
            ang = _compute_angle_3d(norm_coords[:, h, a, :], norm_coords[:, h, b, :], norm_coords[:, h, c, :])
            angles_list.append(ang * present[:, h])
    hand_angles = np.stack(angles_list, axis=-1)  # (N, 30)

    # 6. Fingertip to wrist distances (5 per hand)
    tip_wrist_list = []
    for h in range(2):
        for tip in TIPS:
            d = np.linalg.norm(norm_coords[:, h, tip, :], axis=-1)
            tip_wrist_list.append(d * present[:, h])
    tip_wrist = np.stack(tip_wrist_list, axis=-1)  # (N, 10)

    # 7. Pairwise fingertip distances (10 per hand)
    tip_pairs_list = []
    for h in range(2):
        for i in range(len(TIPS)):
            for j in range(i + 1, len(TIPS)):
                d = np.linalg.norm(norm_coords[:, h, TIPS[i], :] - norm_coords[:, h, TIPS[j], :], axis=-1)
                tip_pairs_list.append(d * present[:, h])
    tip_pairs = np.stack(tip_pairs_list, axis=-1)  # (N, 20)

    # 8. Finger curl extension ratios: tip-to-wrist / mcp-to-wrist (5 per hand)
    curl_ratio_list = []
    for h in range(2):
        for idx, (mcp, tip) in enumerate([(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]):
            d_tip = np.linalg.norm(norm_coords[:, h, tip, :], axis=-1)
            d_mcp = np.linalg.norm(norm_coords[:, h, mcp, :], axis=-1) + EPS
            curl_ratio_list.append((d_tip / d_mcp) * present[:, h])
    curl_ratios = np.stack(curl_ratio_list, axis=-1)  # (N, 10)

    # 9. Inter-hand features (for 2 hands)
    two_mask = (n_hands == 2).astype(np.float32)
    mean_scale = np.where(two_mask > 0, scale.mean(axis=1), 1.0)
    inter_wrist = (coords[:, 1, WRIST, :] - coords[:, 0, WRIST, :]) / mean_scale[:, None]
    inter_wrist_dist = np.linalg.norm(inter_wrist, axis=-1, keepdims=True)
    inter_wrist_feats = np.concatenate([inter_wrist, inter_wrist_dist], axis=-1) * two_mask[:, None]  # (N, 4)

    # 10. Inter-hand fingertip contact matrix: 5 tips x 5 tips = 25 distances
    inter_tip_list = []
    for t0 in TIPS:
        for t1 in TIPS:
            d = np.linalg.norm((coords[:, 0, t0, :] - coords[:, 1, t1, :]) / mean_scale[:, None], axis=-1)
            inter_tip_list.append(d * two_mask)
    inter_tip_dists = np.stack(inter_tip_list, axis=-1)  # (N, 25)

    nh_feat = n_hands[:, None].astype(np.float32)  # (N, 1)

    all_feats = np.concatenate([
        flat_lms,           # 126
        hand_angles,        # 30
        tip_wrist,          # 10
        tip_pairs,          # 20
        curl_ratios,        # 10
        nh_feat,            # 1
        inter_wrist_feats,  # 4
        inter_tip_dists     # 25
    ], axis=1).astype(np.float32)

    return all_feats


# --------------------------------------------------------------------------- legacy feature blocks
def _hand_scale(coords: np.ndarray) -> np.ndarray:
    d = coords[:, :, MIDDLE_MCP, :2] - coords[:, :, WRIST, :2]
    return np.linalg.norm(d, axis=-1)


def _normalised(coords: np.ndarray, present: np.ndarray):
    scale = _hand_scale(coords)
    scale = np.where(scale < EPS, 1.0, scale)
    rel = coords - coords[:, :, WRIST:WRIST + 1, :]
    rel = rel / scale[:, :, None, None]
    rel = rel * present[:, :, None, None]
    return rel, scale


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + EPS)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _engineered(rel: np.ndarray, present: np.ndarray) -> np.ndarray:
    tips = rel[:, :, TIPS, :]
    tip_wrist = np.linalg.norm(tips, axis=-1)
    pairs = [np.linalg.norm(tips[:, :, i] - tips[:, :, j], axis=-1)
             for i in range(5) for j in range(i + 1, 5)]
    pairs = np.stack(pairs, axis=-1)
    curls = np.stack([_angle(rel[:, :, a], rel[:, :, b], rel[:, :, c]) for a, b, c in CURL], axis=-1)
    feats = np.concatenate([tip_wrist, pairs, curls], axis=-1)
    return feats * present[:, :, None]


def _inter_hand(coords: np.ndarray, n_hands: np.ndarray, scale: np.ndarray) -> np.ndarray:
    two = (n_hands == 2)
    mean_scale = np.where(two, scale.mean(axis=1), 1.0)
    mean_scale = np.where(mean_scale < EPS, 1.0, mean_scale)
    d = coords[:, 1, WRIST, :] - coords[:, 0, WRIST, :]
    d = d / mean_scale[:, None]
    dist = np.linalg.norm(d[:, :2], axis=-1, keepdims=True)
    out = np.concatenate([d, dist], axis=1)
    return out * two[:, None]


# --------------------------------------------------------------------------- public API
def build_features(coords: np.ndarray, n_hands: np.ndarray, mode: str = "robust_isl",
                   labels: np.ndarray | None = None) -> np.ndarray:
    """Build feature matrix for hand landmarks."""
    coords = np.asarray(coords, dtype=np.float32)
    n_hands = np.asarray(n_hands, dtype=int)
    n = len(coords)

    if mode == "robust_isl":
        return _extract_robust_features(coords, n_hands, labels)

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
                    out[i, :63] = flat
                else:
                    out[i, 63:] = flat
        return out

    present = (np.arange(2)[None, :] < n_hands[:, None]).astype(np.float32)
    rel, scale = _normalised(coords, present)
    base = [rel.reshape(n, -1), n_hands[:, None].astype(np.float32),
            _inter_hand(coords, n_hands, scale)]
    if mode == "norm":
        return np.concatenate(base, axis=1).astype(np.float32)
    if mode == "norm_eng":
        eng = _engineered(rel, present).reshape(n, -1)
        return np.concatenate(base + [eng], axis=1).astype(np.float32)
    raise ValueError(f"unknown feature mode: {mode} (choose from {FEATURE_MODES})")


def feature_names(mode: str = "robust_isl") -> List[str]:
    """Human-readable feature names."""
    axes = "xyz"
    if mode in ("sorted_raw", "legacy_raw126"):
        return [f"h{h}_lm{j}_{a}" for h in range(2) for j in range(21) for a in axes]
    if mode == "robust_isl":
        names = [f"h{h}_rot_lm{j}_{a}" for h in range(2) for j in range(21) for a in axes]
        joint_names = ["cmc", "mcp", "ip", "mcp", "pip", "dip", "mcp", "pip", "dip", "mcp", "pip", "dip", "mcp", "pip", "dip"]
        finger_names = ["thumb", "thumb", "thumb", "index", "index", "index", "middle", "middle", "middle", "ring", "ring", "ring", "pinky", "pinky", "pinky"]
        for h in range(2):
            names += [f"h{h}_{f}_{j}_angle" for f, j in zip(finger_names, joint_names)]
        tip_names = ["thumb", "index", "middle", "ring", "pinky"]
        for h in range(2):
            names += [f"h{h}_{t}_tip_to_wrist" for t in tip_names]
        for h in range(2):
            names += [f"h{h}_{tip_names[i]}_{tip_names[j]}_tip_dist" for i in range(5) for j in range(i + 1, 5)]
        for h in range(2):
            names += [f"h{h}_{t}_extension_ratio" for t in tip_names]
        names += ["n_hands", "inter_wrist_dx", "inter_wrist_dy", "inter_wrist_dz", "inter_wrist_dist"]
        for t0 in tip_names:
            for t1 in tip_names:
                names.append(f"inter_tip_h0_{t0}_h1_{t1}")
        return names

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
