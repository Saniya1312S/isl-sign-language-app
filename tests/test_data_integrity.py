"""The committed landmark table and the EDA summary must be consistent with each other."""
import json

import numpy as np
import pytest

from isl.config import LANDMARKS_CSV, RESULTS_DIR
from isl.data import COORD_COLS, load_landmarks

pytestmark = pytest.mark.skipif(not LANDMARKS_CSV.exists(), reason="run scripts/extract_landmarks.py first")


@pytest.fixture(scope="module")
def df():
    return load_landmarks()


def test_schema_and_ranges(df):
    assert set(COORD_COLS) <= set(df.columns)
    assert df["n_hands"].isin([0, 1, 2]).all()
    assert set(df["label"]) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert not df[COORD_COLS].isna().any().any()
    present = df[df.n_hands > 0][COORD_COLS].to_numpy()
    assert present.min() > -1.0 and present.max() < 2.0          # normalised image coordinates


def test_missing_hands_have_zero_coordinates(df):
    none = df[df.n_hands == 0][COORD_COLS].to_numpy()
    assert not none.any()


def test_second_hand_block_empty_when_one_hand(df):
    one = df[df.n_hands == 1]
    assert not one[[c for c in COORD_COLS if c.startswith("h1_")]].to_numpy().any()


def test_no_duplicate_files_per_source(df):
    assert not df.duplicated(["source", "label", "file"]).any()


def test_eda_summary_matches_table(df):
    p = RESULTS_DIR / "eda.json"
    if not p.exists():
        pytest.skip("eda.json not generated")
    eda = json.loads(p.read_text())
    assert eda["total_images"] == len(df)
    for src, v in eda["sources"].items():
        g = df[df.source == src]
        assert v["images"] == len(g) and v["images_with_hand"] == int((g.n_hands > 0).sum())
        assert sum(v["per_class"].values()) == len(g)
