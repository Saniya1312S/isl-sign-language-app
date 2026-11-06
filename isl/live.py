"""
Real-time video processing for the browser camera (streamlit-webrtc).

WHY this replaces `cv2.VideoCapture(0)`:
  The camera is opened by the USER'S BROWSER (with a permission prompt) and frames are streamed to
  this processor. `cv2.VideoCapture(0)` opens a camera attached to the SERVER, which
    - fails on any cloud deployment (the server has no camera),
    - fails if another program already holds the camera (the common "camera won't open" error),
    - needed a blocking `while` loop inside Streamlit (the page froze, no clean way to stop).

`SignVideoProcessor.process_bgr()` is a plain function (numpy in, numpy out), so it is unit-tested
without a browser. `recv()` is only a thin adapter for streamlit-webrtc.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Dict, Optional

import cv2
import numpy as np

from .config import DEFAULT_HOLD_FRAMES, DEFAULT_MIN_CONFIDENCE
from .features import hands_to_arrays
from .hands import HandDetector, draw_hands
from .personalise import CalibrationSet
from .predictor import Prediction, SignPredictor
from .spelling import SpellingBuffer

try:                                   # streamlit-webrtc is only needed for the live page
    import av
    from streamlit_webrtc import VideoProcessorBase
except Exception:                      # pragma: no cover
    av = None
    VideoProcessorBase = object

TEAL, WHITE, DARK, AMBER = (166, 184, 20), (255, 255, 255), (30, 24, 17), (11, 158, 245)   # BGR


class SignVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.RLock()
        self.detector: Optional[HandDetector] = None
        self.predictors: Dict[str, SignPredictor] = {}
        self.mode = "alphabet"
        self.buffer = SpellingBuffer(hold_frames=DEFAULT_HOLD_FRAMES)
        self.min_confidence = DEFAULT_MIN_CONFIDENCE
        self.mirror_view = True
        self.last = Prediction(None, 0.0, 0, "no_hand")
        self.fps = 0.0
        self.latency_ms = 0.0
        self.frames = 0
        self._t = deque(maxlen=30)
        self.log = deque(maxlen=500)          # (time, label, confidence) of every TYPED token
        # personalisation: while recording, landmark samples of one letter are collected instead of typing
        self.calibration: Optional[CalibrationSet] = None
        self._rec_letter: Optional[str] = None
        self._rec_left = 0
        self._rec_tick = 0

    # ------------------------------------------------------------------ configuration (UI thread)
    def configure(self, predictors: Dict[str, SignPredictor], mode: str, hold_frames: int,
                  min_confidence: float, mirror_view: bool = True, hold_seconds: Optional[float] = None) -> None:
        with self._lock:
            self.predictors = predictors
            self.mode = mode if mode in predictors else next(iter(predictors), "alphabet")
            self.min_confidence = min_confidence
            self.mirror_view = mirror_view
            for p in predictors.values():
                p.min_confidence = min_confidence
            self.buffer.reconfigure(hold_frames=hold_frames, word_mode=(self.mode == "words"), hold_seconds=hold_seconds)

    def start_recording(self, letter: str, n_samples: int = 12) -> None:
        with self._lock:
            self._rec_letter, self._rec_left, self._rec_tick = letter, int(n_samples), 0

    @property
    def recording(self):
        """(letter, samples_left) while recording, else None."""
        return (self._rec_letter, self._rec_left) if self._rec_letter else None

    # ------------------------------------------------------------------ per-frame work
    def process_bgr(self, img: np.ndarray) -> np.ndarray:
        """Detect -> classify -> update spelling -> return the annotated frame (BGR)."""
        t0 = time.perf_counter()
        if self.detector is None:
            self.detector = HandDetector(static_image_mode=False)
        hands = self.detector.detect_bgr(img)
        with self._lock:
            predictor = self.predictors.get(self.mode)
        pred = predictor.predict_hands(hands) if predictor else Prediction(None, 0.0, len(hands), "no_model")
        if self._rec_letter and self.calibration is not None:
            self.buffer.update(None)                       # do not type while teaching
            if hands:
                self._rec_tick += 1
                if self._rec_tick % 3 == 0:                # every 3rd frame -> samples are not identical twins
                    coords, nh, _ = hands_to_arrays(hands)
                    self.calibration.add(self._rec_letter, coords[0], int(nh[0]))
                    self._rec_left -= 1
                    if self._rec_left <= 0:
                        self._rec_letter = None
            typed = None
        else:
            typed = self.buffer.update(pred.label)
        if typed:
            self.log.append((time.time(), typed, pred.confidence))
        self.last = pred

        out = draw_hands(img, hands)
        if self.mirror_view:
            out = cv2.flip(out, 1)                  # selfie view; text is drawn AFTER flipping
        self._overlay(out, pred)

        now = time.perf_counter()
        self.latency_ms = 0.9 * self.latency_ms + 0.1 * (now - t0) * 1000 if self.frames else (now - t0) * 1000
        self._t.append(now)
        if len(self._t) > 1:
            self.fps = (len(self._t) - 1) / (self._t[-1] - self._t[0])
        self.frames += 1
        return out

    def _overlay(self, out: np.ndarray, pred: Prediction) -> None:
        h, w = out.shape[:2]
        snap = self.buffer.snapshot()
        cv2.rectangle(out, (0, 0), (w, 46), DARK, -1)
        if self._rec_letter:
            cv2.putText(out, f"REC {self._rec_letter}  ({self._rec_left} left) - hold the sign", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2, cv2.LINE_AA)
        elif pred.label:
            cv2.putText(out, f"{pred.label}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEAL, 3, cv2.LINE_AA)
            cv2.putText(out, f"{pred.confidence:.0%}", (12 + 30 * max(len(pred.label), 1) + 20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        else:
            msg = {"no_hand": "show your hand", "low_confidence": "not sure - hold steady",
                   "no_model": "model not loaded"}.get(pred.reason, "")
            cv2.putText(out, msg, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AMBER, 2, cv2.LINE_AA)
        # hold-progress bar along the bottom
        cv2.rectangle(out, (0, h - 10), (w, h), DARK, -1)
        cv2.rectangle(out, (0, h - 10), (int(w * snap.progress), h), TEAL, -1)

    # ------------------------------------------------------------------ streamlit-webrtc adapter
    def recv(self, frame):                                  # pragma: no cover  (needs a browser)
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(self.process_bgr(img), format="bgr24")

    def stats(self) -> dict:
        return {"fps": round(self.fps, 1), "latency_ms": round(self.latency_ms, 1), "frames": self.frames}


# --------------------------------------------------------------------------- single images
STATIC_LOCK = threading.Lock()      # MediaPipe objects are not thread-safe; Streamlit sessions run in threads


def analyse_image(img_bgr: np.ndarray, detector: HandDetector, predictor: SignPredictor):
    """One still image -> (annotated BGR image, Prediction). Used by the snapshot and upload tabs."""
    with STATIC_LOCK:
        hands = detector.detect_bgr(img_bgr)
    pred = predictor.predict_hands(hands)
    return draw_hands(img_bgr, hands), pred
