"""
Hand detection with MediaPipe Hands.

MediaPipe is pinned to 0.10.21 (see requirements.txt): it is the last release that
ships the hand models inside the wheel (works offline) and still provides
`mediapipe.solutions.hands`.

Every detected hand is returned as a `Hand` with 21 (x, y, z) landmarks.
x and y are normalised to the image width/height (0..1); z is relative depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

try:  # imported lazily so the dashboard still works if MediaPipe is missing
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
        if mp is None or not hasattr(mp, "solutions"):
            raise RuntimeError(
                "MediaPipe with `solutions.hands` is required. Install the pinned version: "
                "pip install mediapipe==0.10.21 (Python 3.9-3.12).")
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
        return self.detect_rgb(bgr[:, :, ::-1])

    def close(self) -> None:
        self._hands.close()


def draw_hands(image_bgr: np.ndarray, hands: List[Hand]) -> np.ndarray:
    """Draw the 21-point skeleton on a copy of a BGR image."""
    out = image_bgr.copy()
    if mp is None:
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
