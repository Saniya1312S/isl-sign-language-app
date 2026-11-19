"""
Unit tests for src.feature_extraction:
- Feature dimensionalities
- Coordinate and scale normalization invariance
- Handling of 1-hand vs 2-hands inputs
- Finger invariant angles and ratios
"""
import numpy as np
import pytest

from src.feature_extraction import (
    build_features,
    extract_features_from_image,
    extract_norm_eng_features,
)


def make_synthetic_hand(wrist=(0.5, 0.5, 0.0), scale=0.1):
    # 21 landmarks
    hand = np.zeros((21, 3), dtype=np.float32)
    wx, wy, wz = wrist
    for i in range(21):
        hand[i] = [wx + i * scale * 0.05, wy - i * scale * 0.05, wz]
    return hand


def test_feature_dimensions():
    coords = np.zeros((3, 2, 21, 3), dtype=np.float32)
    coords[0, 0] = make_synthetic_hand((0.3, 0.5, 0.0))
    coords[1, 0] = make_synthetic_hand((0.3, 0.5, 0.0))
    coords[1, 1] = make_synthetic_hand((0.7, 0.5, 0.0))
    n_hands = np.array([1, 2, 0])

    X = build_features(coords, n_hands, mode="norm_eng")
    assert X.shape == (3, 171)
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()


def test_translation_and_scale_invariance():
    # Base hand
    hand1 = make_synthetic_hand(wrist=(0.4, 0.4, 0.0), scale=0.1)
    # Translated and scaled hand
    hand2 = make_synthetic_hand(wrist=(0.6, 0.8, 0.1), scale=0.25)

    coords = np.zeros((2, 2, 21, 3), dtype=np.float32)
    coords[0, 0] = hand1
    coords[1, 0] = hand2
    n_hands = np.array([1, 1])

    X = build_features(coords, n_hands, mode="norm_eng")
    # Features for single hand (finger angles and relative landmark positions normalized by scale)
    # should be extremely close despite translation and scale change
    diff = np.abs(X[0, :63] - X[1, :63]).max()
    assert diff < 1e-3, f"Features not translation/scale invariant, max diff: {diff}"


def test_extract_features_from_image_blank():
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    res = extract_features_from_image(blank)
    assert res["status"] == "no_hand_detected"
    assert res["features"] is None
    assert res["n_hands"] == 0
    assert res["annotated_image"].shape == blank.shape
