"""
Feature engineering and unified image feature extraction.
Guarantees identical feature formulation between training and prediction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .preprocessing import (
    DetectedHand,
    MediaPipeHandDetector,
    canonical_order,
    draw_landmarks_on_image,
    mirror,
)

WRIST = 0
MIDDLE_MCP = 9
TIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky
CURL_TRIPLES = ((2, 3, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
EPS = 1e-6

# Default detector cache
_GLOBAL_DETECTOR: Optional[MediaPipeHandDetector] = None


def get_detector() -> MediaPipeHandDetector:
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = MediaPipeHandDetector(static_image_mode=True)
    return _GLOBAL_DETECTOR


def compute_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at joint b formed by vectors (a - b) and (c - b)."""
    v1 = a - b
    v2 = c - b
    v1_norm = np.linalg.norm(v1, axis=-1, keepdims=True) + EPS
    v2_norm = np.linalg.norm(v2, axis=-1, keepdims=True) + EPS
    cos = np.sum((v1 / v1_norm) * (v2 / v2_norm), axis=-1)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def extract_norm_eng_features(coords: np.ndarray, n_hands: np.ndarray) -> np.ndarray:
    """
    171-dimensional feature set:
    1. Wrist-centered coordinates (126 dims)
    2. Scale normalization by palm size
    3. Hand count (1 dim)
    4. Inter-wrist geometry (4 dims)
    5. Fingertip-to-wrist distances (10 dims)
    6. Pairwise fingertip distances (20 dims)
    7. Finger curl angles (10 dims)
    """
    n = len(coords)
    present = (np.arange(2)[None, :] < n_hands[:, None]).astype(np.float32)

    # 1. Wrist subtraction & Scale normalization
    d_mcp = coords[:, :, MIDDLE_MCP, :2] - coords[:, :, WRIST, :2]
    scale = np.linalg.norm(d_mcp, axis=-1)
    scale = np.where(scale < EPS, 1.0, scale)

    rel = coords - coords[:, :, WRIST:WRIST + 1, :]
    rel = rel / scale[:, :, None, None]
    rel = rel * present[:, :, None, None]

    flat_rel = rel.reshape(n, -1)  # 126
    nh_feat = n_hands[:, None].astype(np.float32)  # 1

    # Inter-hand geometry
    two = (n_hands == 2)
    mean_scale = np.where(two, scale.mean(axis=1), 1.0)
    mean_scale = np.where(mean_scale < EPS, 1.0, mean_scale)
    inter_d = (coords[:, 1, WRIST, :] - coords[:, 0, WRIST, :]) / mean_scale[:, None]
    inter_dist = np.linalg.norm(inter_d[:, :2], axis=-1, keepdims=True)
    inter_wrist = np.concatenate([inter_d, inter_dist], axis=1) * two[:, None]  # 4

    # Engineered finger features
    tips = rel[:, :, TIPS, :]
    tip_wrist = np.linalg.norm(tips, axis=-1)  # (N, 2, 5)

    pairs = [np.linalg.norm(tips[:, :, i] - tips[:, :, j], axis=-1)
             for i in range(5) for j in range(i + 1, 5)]
    pairs = np.stack(pairs, axis=-1)  # (N, 2, 10)

    curls = np.stack([compute_angle_3d(rel[:, :, a], rel[:, :, b], rel[:, :, c])
                      for a, b, c in CURL_TRIPLES], axis=-1)  # (N, 2, 5)

    finger_feats = np.concatenate([tip_wrist, pairs, curls], axis=-1) * present[:, :, None]
    flat_finger = finger_feats.reshape(n, -1)  # 40

    features = np.concatenate([flat_rel, nh_feat, inter_wrist, flat_finger], axis=1).astype(np.float32)
    return features


def build_features(coords: np.ndarray, n_hands: np.ndarray,
                   mode: str = "norm_eng", labels: Optional[np.ndarray] = None) -> np.ndarray:
    """Build standardized feature matrix for landmark arrays."""
    coords = np.asarray(coords, dtype=np.float32)
    n_hands = np.asarray(n_hands, dtype=int)

    if mode == "norm_eng":
        return extract_norm_eng_features(coords, n_hands)
    else:
        # Fallback or robust_isl from isl.features if needed
        from isl.features import build_features as isl_build_features
        return isl_build_features(coords, n_hands, mode=mode, labels=labels)


def extract_features_from_image(image: np.ndarray, detector: Optional[MediaPipeHandDetector] = None,
                                mode: str = "norm_eng") -> Dict[str, Any]:
    """
    Mandatory Phase 4 Reusable Feature Extraction Function.

    Input:
        image: np.ndarray (BGR or RGB)
        detector: optional MediaPipeHandDetector instance
        mode: feature mode ('norm_eng')

    Returns:
        {
            "landmarks": list of DetectedHand,
            "features": np.ndarray of shape (1, n_features) or None,
            "n_hands": int (0, 1, or 2),
            "status": "success" | "no_hand_detected" | "invalid_image",
            "annotated_image": np.ndarray (annotated BGR image)
        }
    """
    if image is None or image.size == 0:
        return {
            "landmarks": [],
            "features": None,
            "n_hands": 0,
            "status": "invalid_image",
            "annotated_image": np.zeros((200, 200, 3), dtype=np.uint8)
        }

    det = detector if detector is not None else get_detector()
    hands = det.detect_bgr(image)
    annotated = draw_landmarks_on_image(image, hands)

    if not hands:
        return {
            "landmarks": [],
            "features": None,
            "n_hands": 0,
            "status": "no_hand_detected",
            "annotated_image": annotated
        }

    # Format arrays (1, 2, 21, 3)
    coords = np.zeros((1, 2, 21, 3), dtype=np.float32)
    labels = np.full((1, 2), "", dtype=object)
    for i, h in enumerate(hands[:2]):
        coords[0, i] = h.landmarks
        labels[0, i] = h.label

    n_hands_arr = np.array([len(hands)], dtype=int)
    feat_vector = build_features(coords, n_hands_arr, mode=mode, labels=labels)

    return {
        "landmarks": hands,
        "features": feat_vector,
        "n_hands": len(hands),
        "status": "success",
        "annotated_image": annotated
    }
