"""Load the landmark table produced by scripts/extract_landmarks.py."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import LANDMARKS_CSV
from .features import canonical_order

COORD_COLS = [f"h{h}_{j}_{a}" for h in range(2) for j in range(21) for a in "xyz"]


def load_landmarks(path=LANDMARKS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[])
    df["h0_label"] = df["h0_label"].astype(str)
    df["h1_label"] = df["h1_label"].astype(str)
    return df


def to_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """DataFrame -> dict(coords (N,2,21,3), n_hands (N,), labels (N,2), y (N,))."""
    coords = df[COORD_COLS].to_numpy(dtype=np.float32).reshape(-1, 2, 21, 3)
    n_hands = df["n_hands"].to_numpy(dtype=int)
    labels = df[["h0_label", "h1_label"]].to_numpy(dtype=object)
    return {"coords": canonical_order(coords, n_hands), "n_hands": n_hands,
            "labels": labels, "y": df["label"].astype(str).to_numpy()}


def blocks(df: pd.DataFrame, size: int = 200) -> np.ndarray:
    """Group id = class + frame block. Consecutive frames come from the same capture, so whole
    blocks are kept together when splitting (prevents near-duplicate leakage)."""
    return (df["label"].astype(str) + "_" + (df["frame_idx"] // size).astype(str)).to_numpy()
