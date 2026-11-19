"""
Data loading and group-aware train/val/test splitting.
Prevents video-frame leakage by holding out entire consecutive frame blocks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "landmarks_alphabet.csv.gz"
COORD_COLS = [f"h{h}_{j}_{a}" for h in range(2) for j in range(21) for a in ("x", "y", "z")]
BLOCK_SIZE = 200
RANDOM_STATE = 42


def load_raw_dataframe(path: Path | str | None = None) -> pd.DataFrame:
    """Load the full raw landmarks CSV."""
    p = Path(path) if path else DEFAULT_DATA_PATH
    if not p.exists():
        raise FileNotFoundError(f"Landmarks dataset not found at: {p}")
    return pd.read_csv(p)


def filter_valid_hands(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out rows where no hands were detected."""
    return df[df["n_hands"] > 0].reset_index(drop=True)


def to_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extract structured arrays from DataFrame:
    - coords: (N, 2, 21, 3) float32
    - n_hands: (N,) int32
    - labels: (N, 2) string object ('Left'/'Right'/empty)
    - y: (N,) string class labels ('A'-'Z')
    """
    n = len(df)
    coords = df[COORD_COLS].to_numpy(dtype=np.float32).reshape(n, 2, 21, 3)
    n_hands = df["n_hands"].to_numpy(dtype=np.int32)
    h0 = df["h0_label"].fillna("").to_numpy(dtype=object)
    h1 = df["h1_label"].fillna("").to_numpy(dtype=object)
    labels = np.column_stack([h0, h1])
    y = df["label"].astype(str).to_numpy()
    return {"coords": coords, "n_hands": n_hands, "labels": labels, "y": y}


def assign_frame_blocks(df: pd.DataFrame, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """
    Assign group IDs to consecutive frame blocks within each class and source.
    Ensures frames recorded sequentially share a group identifier.
    """
    groups = np.zeros(len(df), dtype=int)
    group_counter = 0
    for (src, label), g in df.groupby(["source", "label"], sort=False):
        # Order by frame_idx or file
        sort_col = "frame_idx" if "frame_idx" in g.columns else "file"
        order = g.sort_values(sort_col).index
        n_items = len(order)
        n_blocks = max(1, int(np.ceil(n_items / block_size)))
        for b_idx in range(n_blocks):
            start = b_idx * block_size
            end = min(start + block_size, n_items)
            block_indices = order[start:end]
            groups[block_indices] = group_counter
            group_counter += 1
    return groups


def create_stratified_block_split(df: pd.DataFrame, val_size: float = 0.15,
                                  test_size: float = 0.15,
                                  random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Hold out entire consecutive frame blocks separately inside EVERY letter.
    Guarantees:
    - Exactly zero leakage between train, val, and test.
    - All 26 alphabet classes exist in train, val, and test.
    - Train is ~70%, Val is ~15%, Test is ~15%.
    """
    rng = np.random.RandomState(random_state)
    train_parts, val_parts, test_parts = [], [], []

    for label, g in df.groupby("label"):
        # Sort chronologically by frame_idx
        g = g.sort_values("frame_idx").reset_index(drop=True)
        n = len(g)
        n_blocks = max(3, int(np.ceil(n / BLOCK_SIZE)))
        block_indices = np.array_split(np.arange(n), n_blocks)

        perm = rng.permutation(n_blocks)
        n_test_blocks = max(1, int(np.round(n_blocks * test_size)))
        n_val_blocks = max(1, int(np.round(n_blocks * val_size)))

        test_b = perm[:n_test_blocks]
        val_b = perm[n_test_blocks:n_test_blocks + n_val_blocks]
        train_b = perm[n_test_blocks + n_val_blocks:]

        if len(train_b) == 0:
            # Fallback if too few blocks
            train_b = val_b[:1]
            val_b = val_b[1:]

        test_idx = np.concatenate([block_indices[b] for b in test_b])
        val_idx = np.concatenate([block_indices[b] for b in val_b])
        train_idx = np.concatenate([block_indices[b] for b in train_b])

        test_parts.append(g.iloc[test_idx])
        val_parts.append(g.iloc[val_idx])
        train_parts.append(g.iloc[train_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, val_df, test_df
