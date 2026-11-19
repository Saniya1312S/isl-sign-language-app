"""
Few-shot personalisation: retrain the classifier with a few examples recorded by the user.

Why: the main study (Dashboard > Evaluation honesty) shows that a model trained on one studio dataset scores
only ~30% on other people / backgrounds. `scripts/adaptation_study.py` shows that a handful of examples from the
NEW setting recovers a lot (1 example per letter: 30% -> 42%, 8 per letter: 30% -> 56%) without forgetting the
studio data (accuracy stays ~99%).

How it works (simple enough to explain in an interview):
    1. The user records a few frames of each letter they care about  ->  landmark samples (not images!).
    2. We take the shipped training data, add the user's samples (mirror-augmented and repeated so that they
       count more than a single studio frame), and re-fit the same kind of classifier. Logistic Regression
       trains in a few seconds on a CPU.
    3. The personal model is used instead of the shared model for this browser session. Nothing is uploaded or
       stored on the server; the samples can be downloaded as a small CSV and loaded again next time.
"""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import LANDMARKS_CSV, RANDOM_STATE
from .data import COORD_COLS, load_landmarks, to_arrays
from .features import build_features, canonical_order, mirror
from .model_io import ModelBundle

USER_REPEAT = 10          # every user sample counts as this many studio frames


@dataclass
class CalibrationSet:
    """Landmark samples recorded by the user, per letter. Thread-safe (filled by the video thread)."""
    samples: Dict[str, List[Tuple[np.ndarray, int]]] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def add(self, letter: str, coords: np.ndarray, n_hands: int) -> None:
        with self._lock:
            self.samples.setdefault(letter, []).append((np.asarray(coords, dtype=np.float32).copy(), int(n_hands)))

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in sorted(self.samples.items())}

    def total(self) -> int:
        return sum(self.counts().values())

    def clear(self, letter: str | None = None) -> None:
        with self._lock:
            if letter is None:
                self.samples.clear()
            else:
                self.samples.pop(letter, None)

    def arrays(self):
        """-> coords (N,2,21,3), n_hands (N,), labels (N,)"""
        with self._lock:
            items = [(k, c, n) for k, v in self.samples.items() for c, n in v]
        if not items:
            return np.zeros((0, 2, 21, 3), np.float32), np.zeros(0, int), np.array([], dtype=object)
        return (np.stack([c for _, c, _ in items]), np.array([n for _, _, n in items]),
                np.array([k for k, _, _ in items], dtype=object))

    # ---- save / load as a small CSV (landmarks only - no images, no faces)
    def to_csv(self) -> bytes:
        coords, nh, y = self.arrays()
        df = pd.DataFrame(coords.reshape(len(coords), -1), columns=COORD_COLS)
        df.insert(0, "n_hands", nh)
        df.insert(0, "letter", y)
        return df.to_csv(index=False).encode("utf-8")

    def load_csv(self, data: bytes) -> int:
        df = pd.read_csv(io.BytesIO(data))
        if not {"letter", "n_hands", *COORD_COLS} <= set(df.columns):
            raise ValueError("This file is not a calibration CSV exported by this app.")
        coords = canonical_order(df[COORD_COLS].to_numpy(np.float32).reshape(-1, 2, 21, 3), df["n_hands"].to_numpy(int))
        for letter, c, n in zip(df["letter"].astype(str), coords, df["n_hands"].astype(int)):
            self.add(letter, c, n)
        return len(df)


def base_training_set(cap_per_class: int = 250, path=LANDMARKS_CSV, mode: str = "norm_eng"):
    """The shipped training data as (X, y): studio frames (capped, spread over the frame order) + external images."""
    df = load_landmarks(path)
    df = df[df.n_hands > 0]
    parts = []
    for (src, label), g in df.groupby(["source", "label"]):
        g = g.sort_values("frame_idx")
        if src == "Indian" and len(g) > cap_per_class:
            g = g.iloc[np.unique(np.linspace(0, len(g) - 1, cap_per_class).round().astype(int))]
        parts.append(g)
    d = pd.concat(parts)
    a = to_arrays(d)
    X = np.vstack([build_features(a["coords"], a["n_hands"], mode), build_features(mirror(a["coords"], a["n_hands"]), a["n_hands"], mode)])
    return X, np.concatenate([a["y"], a["y"]])


def personalise(base: ModelBundle, base_X: np.ndarray, base_y: np.ndarray, cal: CalibrationSet,
                repeat: int = USER_REPEAT) -> ModelBundle:
    """Return a new ModelBundle trained on shipped data + the user's samples."""
    if base.feature_mode != "norm_eng":
        raise ValueError("Personalisation needs a model that uses the 'norm_eng' features.")
    coords, nh, y = cal.arrays()
    if len(y) == 0:
        raise ValueError("Record at least one sample first.")
    Xu = np.vstack([build_features(coords, nh, base.feature_mode), build_features(mirror(coords, nh), nh, base.feature_mode)])
    yu = np.concatenate([y, y])
    X = np.vstack([base_X] + [Xu] * repeat)
    ys = np.concatenate([base_y] + [yu] * repeat)
    model = clone(base.model).fit(X, ys)
    meta = dict(base.meta)
    meta.update({"personalised": True, "user_samples": cal.counts(), "user_repeat": repeat, "random_state": RANDOM_STATE})
    return ModelBundle(model, meta)
