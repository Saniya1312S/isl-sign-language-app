import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Set environment variables for sandbox and clean imports
os.environ["KERAS_HOME"] = str(Path(__file__).resolve().parent.parent / ".keras")
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".matplotlib")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from isl.data import load_landmarks, to_arrays

WRIST = 0
MIDDLE_MCP = 9
TIPS = (4, 8, 12, 16, 20)
EPS = 1e-6

# Joint triples: (base, joint, tip)
JOINT_ANGLES = [
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
]

ABDUCTION_PAIRS = [
    (1, 2, 5, 6),   # thumb to index
    (5, 6, 9, 10),  # index to middle
    (9, 10, 13, 14),# middle to ring
    (13, 14, 17, 18)# ring to pinky
]

def compute_angle_3d(a, b, c):
    """Angle at joint b formed by vectors (a - b) and (c - b). Shape (N, ...)."""
    v1 = a - b
    v2 = c - b
    v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True) + EPS
    v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True) + EPS
    cos = np.sum((v1 / v1_norm) * (v2 / v2_norm), axis=-1)
    return np.arccos(np.clip(cos, -1.0, 1.0))

def extract_robust_features(coords, n_hands, labels=None):
    """
    coords: shape (N, 2, 21, 3)
    n_hands: shape (N,)
    labels: shape (N, 2) strings like 'Left', 'Right', ''
    """
    N = len(coords)
    present = (np.arange(2)[None, :] < n_hands[:, None]).astype(np.float32)  # (N, 2)
    
    # 1. Wrist subtraction (translation invariance)
    # rel: (N, 2, 21, 3)
    rel = coords - coords[:, :, WRIST:WRIST+1, :]
    
    # 2. Scale normalization: distance wrist -> middle MCP
    scale = np.linalg.norm(rel[:, :, MIDDLE_MCP, :2], axis=-1)  # (N, 2)
    scale = np.where(scale < EPS, 1.0, scale)
    
    # 3. In-plane rotation alignment: rotate so wrist -> middle MCP points along (0, -1)
    # dx, dy of middle MCP relative to wrist
    v = rel[:, :, MIDDLE_MCP, :2]  # (N, 2, 2)
    current_angle = np.arctan2(v[:, :, 1], v[:, :, 0])  # (N, 2)
    target_angle = -np.pi / 2.0
    rot_angle = target_angle - current_angle  # (N, 2)
    
    cos_a = np.cos(rot_angle)[:, :, None]  # (N, 2, 1)
    sin_a = np.sin(rot_angle)[:, :, None]  # (N, 2, 1)
    
    # Rotate (x, y) coordinates
    rx = rel[:, :, :, 0] * cos_a - rel[:, :, :, 1] * sin_a
    ry = rel[:, :, :, 0] * sin_a + rel[:, :, :, 1] * cos_a
    rz = rel[:, :, :, 2]
    
    # Scale normalized coordinates
    norm_coords = np.stack([rx / scale[:, :, None], ry / scale[:, :, None], rz / scale[:, :, None]], axis=-1)
    
    # 4. Handedness normalization for single-hand signs:
    # If exactly 1 hand and it's labeled 'Left', mirror x -> -x so left and right hands match
    if labels is not None:
        single_left = (n_hands == 1) & (labels[:, 0] == 'Left')
        norm_coords[single_left, 0, :, 0] = -norm_coords[single_left, 0, :, 0]
        
    norm_coords = norm_coords * present[:, :, None, None]
    
    # Flatten landmarks: (N, 2 * 21 * 3) = 126
    flat_lms = norm_coords.reshape(N, -1)
    
    # 5. Joint angles: 15 per hand = 30
    angles_list = []
    for h in range(2):
        for (a, b, c) in JOINT_ANGLES:
            ang = compute_angle_3d(norm_coords[:, h, a, :], norm_coords[:, h, b, :], norm_coords[:, h, c, :])
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

if __name__ == "__main__":
    t0 = time.time()
    df = load_landmarks()
    a = to_arrays(df)
    X = extract_robust_features(a["coords"], a["n_hands"], a["labels"])
    print(f"Extracted {X.shape} features in {time.time() - t0:.2f}s")
    print(f"Any NaN: {np.isnan(X).any()}")
    
    # Quick model check
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    
    # Split train/test
    X_tr, X_te, y_tr, y_te = train_test_split(X, a["y"], test_size=0.2, random_state=42, stratify=a["y"])
    print("Training HistGradientBoosting...")
    t1 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42).fit(X_tr, y_tr)
    y_pred = hgb.predict(X_te)
    print(f"HistGradientBoosting Accuracy: {accuracy_score(y_te, y_pred):.4f} in {time.time() - t1:.2f}s")
