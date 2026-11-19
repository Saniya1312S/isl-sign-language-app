"""
Preprocessing and hand detection using MediaPipe Hands.
Provides:
- Hand detection & landmark extraction
- Canonical spatial ordering (wrist-x sorted)
- Mirror symmetry augmentation
- Skeleton drawing for visual debugging
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Safe environment paths for MediaPipe/TensorFlow compatibility in isolated sandboxes
if "KERAS_HOME" not in os.environ:
    os.environ["KERAS_HOME"] = tempfile.gettempdir()
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

try:
    import mediapipe as mp
except Exception:
    mp = None

EPS = 1e-6
WRIST = 0
MIDDLE_MCP = 9

# MediaPipe Hand Connections for visual debugging
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # pinky & palm
]


@dataclass
class DetectedHand:
    landmarks: np.ndarray       # shape (21, 3), normalized x, y, z in [0, 1]
    label: str                  # "Left" or "Right" as reported by MediaPipe
    score: float                # confidence score
    raw_landmarks: object = None
    detection_method: str = "standard"  # "standard" or "enhanced_padded"

    @property
    def wrist_x(self) -> float:
        return float(self.landmarks[WRIST, 0])


class MediaPipeHandDetector:
    """Wrapper around mp.solutions.hands.Hands with multi-stage fallback."""

    def __init__(self, static_image_mode: bool = True, max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        if mp is None or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise RuntimeError("MediaPipe with `solutions.hands` is required. Please install mediapipe.")
        self.max_num_hands = max_num_hands
        self._hands_std = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # Enhanced fallback detector with lower confidence threshold for tightly cropped / challenging images
        self._hands_fallback = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=0.20,
            min_tracking_confidence=0.20
        )

    def detect_bgr(self, bgr_image: np.ndarray, enable_fallback: bool = True) -> List[DetectedHand]:
        """Detect hands in a BGR image, using multi-stage context fallback if needed."""
        if bgr_image is None or bgr_image.size == 0:
            return []
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return self.detect_rgb(rgb, enable_fallback=enable_fallback)

    def detect_rgb(self, rgb_image: np.ndarray, enable_fallback: bool = True) -> List[DetectedHand]:
        """Detect hands in an RGB image."""
        if rgb_image is None or rgb_image.size == 0:
            return []

        # 1. Standard detection attempt on raw image
        res = self._hands_std.process(np.ascontiguousarray(rgb_image))
        hands: List[DetectedHand] = []
        if res.multi_hand_landmarks:
            handedness = res.multi_handedness or [None] * len(res.multi_hand_landmarks)
            for lms, h in zip(res.multi_hand_landmarks, handedness):
                coords = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
                label = h.classification[0].label if (h and h.classification) else ""
                score = float(h.classification[0].score) if (h and h.classification) else 1.0
                hands.append(DetectedHand(landmarks=coords, label=label, score=score,
                                          raw_landmarks=lms, detection_method="standard"))
            # Canonical order: sort by wrist x-coordinate (image-left hand first)
            hands = sorted(hands, key=lambda h: h.wrist_x)[:self.max_num_hands]
            return hands

        if not enable_fallback:
            return []

        # 2. Enhanced Fallback: Contextual gray padding to restore palm detector margin
        H_orig, W_orig = rgb_image.shape[:2]
        pad = max(16, int(min(H_orig, W_orig) * 0.18))
        padded = cv2.copyMakeBorder(rgb_image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        H_pad, W_pad = padded.shape[:2]

        res_fb = self._hands_fallback.process(np.ascontiguousarray(padded))
        if res_fb.multi_hand_landmarks:
            handedness = res_fb.multi_handedness or [None] * len(res_fb.multi_hand_landmarks)
            for lms, h in zip(res_fb.multi_hand_landmarks, handedness):
                coords = []
                for p in lms.landmark:
                    x_pix = p.x * W_pad - pad
                    y_pix = p.y * H_pad - pad
                    x_orig = float(np.clip(x_pix / W_orig, 0.0, 1.0))
                    y_orig = float(np.clip(y_pix / H_orig, 0.0, 1.0))
                    coords.append([x_orig, y_orig, p.z])
                coords_arr = np.array(coords, dtype=np.float32)
                label = h.classification[0].label if (h and h.classification) else ""
                score = float(h.classification[0].score) if (h and h.classification) else 1.0
                hands.append(DetectedHand(landmarks=coords_arr, label=label, score=score,
                                          raw_landmarks=lms, detection_method="enhanced_padded"))
            # Canonical order: sort by wrist x-coordinate (image-left hand first)
            hands = sorted(hands, key=lambda h: h.wrist_x)[:self.max_num_hands]
            return hands

        return []


def canonical_order(coords: np.ndarray, n_hands: np.ndarray) -> np.ndarray:
    """Sort hands by wrist x-coordinate (image-left hand first)."""
    out = np.zeros_like(coords)
    for i in range(len(coords)):
        n = int(n_hands[i])
        if n == 0:
            continue
        order = np.argsort(coords[i, :n, WRIST, 0])
        out[i, :n] = coords[i, :n][order]
    return out


def mirror(coords: np.ndarray, n_hands: np.ndarray,
           labels: Optional[np.ndarray] = None) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric mirror reflection:
    1. Invert x -> 1.0 - x.
    2. Swap 'Left' <-> 'Right' handedness labels.
    3. Reorder hands so canonical image-left hand remains at index 0.
    Satisfies mirror(mirror(x)) == x.
    """
    flipped = coords.copy()
    for i in range(len(flipped)):
        n = int(n_hands[i])
        flipped[i, :n, :, 0] = 1.0 - flipped[i, :n, :, 0]
    out_coords = canonical_order(flipped, n_hands)

    if labels is None:
        return out_coords

    out_labels = np.full_like(labels, fill_value="", dtype=object)
    for i in range(len(labels)):
        n = int(n_hands[i])
        if n == 0:
            continue
        order = np.argsort(flipped[i, :n, WRIST, 0])
        swapped = [("Right" if l == "Left" else ("Left" if l == "Right" else str(l))) for l in labels[i, :n]]
        for j, idx in enumerate(order):
            out_labels[i, j] = swapped[idx]

    dtype_out = labels.dtype if labels.dtype.kind in ("U", "S", "O") and labels.dtype != object else object
    return out_coords, out_labels.astype(dtype_out)


def draw_landmarks_on_image(image: np.ndarray, hands: Sequence[DetectedHand]) -> np.ndarray:
    """Render 21 skeletal landmarks on an image copy for visual debugging."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    palette = [
        {"bone": (20, 200, 240), "joint": (255, 255, 255)}, # Hand 0: Teal/Yellow
        {"bone": (240, 180, 40), "joint": (255, 255, 255)}, # Hand 1: Cyan/Amber
    ]

    for h_idx, hand in enumerate(hands[:2]):
        colors = palette[h_idx % len(palette)]
        # Pixel coordinates
        pts = [(int(np.clip(p[0] * w, 0, w - 1)), int(np.clip(p[1] * h, 0, h - 1))) for p in hand.landmarks]

        # Draw bones
        for start, end in HAND_CONNECTIONS:
            cv2.line(annotated, pts[start], pts[end], colors["bone"], 2, cv2.LINE_AA)

        # Draw joint circles
        for idx, (px, py) in enumerate(pts):
            radius = 5 if idx in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(annotated, (px, py), radius, colors["joint"], -1, cv2.LINE_AA)
            cv2.circle(annotated, (px, py), radius, (0, 0, 0), 1, cv2.LINE_AA)

    return annotated
