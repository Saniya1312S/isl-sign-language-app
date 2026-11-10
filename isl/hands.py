"""
Hand detection with MediaPipe Hands.

Every detected hand is returned as a `Hand` with 21 (x, y, z) landmarks.
x and y are normalized to the image width/height (0..1); z is relative depth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# Safeguard environment paths
_base_dir = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(_base_dir / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(_base_dir / ".matplotlib"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None

from .config import MAX_HANDS, N_LANDMARKS


@dataclass
class Hand:
    landmarks: np.ndarray        # shape (21, 3)
    label: str                   # "Left" / "Right" as reported by MediaPipe
    score: float                 # handedness confidence
    raw: object = None           # MediaPipe landmark object (used only for drawing)

    @property
    def wrist_x(self) -> float:
        return float(self.landmarks[0, 0])


class HandDetector:
    """Thin wrapper around mp.solutions.hands.Hands."""

    def __init__(self, static_image_mode: bool = False, max_num_hands: int = MAX_HANDS,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1):
        if mp is None or not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise RuntimeError(
                "MediaPipe with `solutions.hands` is required. Please install mediapipe."
            )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect_rgb(self, rgb: np.ndarray) -> List[Hand]:
        """Detect hands in an RGB uint8 image (H, W, 3)."""
        res = self._hands.process(np.ascontiguousarray(rgb))
        hands: List[Hand] = []
        if res.multi_hand_landmarks:
            handed = res.multi_handedness or [None] * len(res.multi_hand_landmarks)
            for lms, h in zip(res.multi_hand_landmarks, handed):
                arr = np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
                if arr.shape != (N_LANDMARKS, 3):
                    continue
                cls = h.classification[0] if h is not None else None
                hands.append(Hand(arr, cls.label if cls else "Right",
                                  float(cls.score) if cls else 0.0, lms))
        return hands

    def detect_bgr(self, bgr: np.ndarray) -> List[Hand]:
        if bgr is None or bgr.size == 0:
            return []
        return self.detect_rgb(bgr[:, :, ::-1])

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()


def draw_hands(image_bgr: np.ndarray, hands: List[Hand]) -> np.ndarray:
    """Draw the 21-point skeleton on a copy of a BGR image."""
    out = image_bgr.copy()
    if mp is None or not hasattr(mp, "solutions"):
        return out
    drawing = mp.solutions.drawing_utils
    style = mp.solutions.drawing_styles
    for h in hands:
        if h.raw is not None:
            drawing.draw_landmarks(
                out, h.raw, mp.solutions.hands.HAND_CONNECTIONS,
                style.get_default_hand_landmarks_style(),
                style.get_default_hand_connections_style())
    return out
