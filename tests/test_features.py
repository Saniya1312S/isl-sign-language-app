"""Feature engineering: shapes, invariance and the mirror / ordering logic."""
import numpy as np
import pytest

from isl.features import (FEATURE_MODES, build_features, canonical_order, feature_names, mirror)


def fake_hand(seed=0, x0=0.5, y0=0.5, size=0.2):
    rng = np.random.RandomState(seed)
    pts = rng.rand(21, 3).astype(np.float32) * 0.5
    pts[0] = 0                                     # wrist at origin
    pts[9] = [0, -1, 0]                            # middle-finger base straight "up", length 1
    out = pts * size
    out[:, 0] += x0
    out[:, 1] += y0
    return out


def batch(n_hands_list):
    coords = np.zeros((len(n_hands_list), 2, 21, 3), dtype=np.float32)
    for i, n in enumerate(n_hands_list):
        for h in range(n):
            coords[i, h] = fake_hand(seed=i * 3 + h, x0=0.3 + 0.4 * h)
    return coords, np.array(n_hands_list)


@pytest.mark.parametrize("mode", FEATURE_MODES)
def test_shapes_match_names_and_have_no_nan(mode):
    coords, nh = batch([1, 2, 1, 2])
    labels = np.array([["Left", ""], ["Left", "Right"], ["Right", ""], ["Right", "Left"]], dtype=object)
    X = build_features(coords, nh, mode, labels)
    assert X.shape == (4, len(feature_names(mode)))
    assert not np.isnan(X).any()


def test_normalised_features_ignore_position_and_size():
    coords, nh = batch([1, 2])
    moved = coords.copy()
    for i in range(2):
        moved[i, :nh[i], :, :2] = moved[i, :nh[i], :, :2] * 0.6 + 0.15
        moved[i, :nh[i], :, 2] *= 0.6
    a, b = build_features(coords, nh, "norm_eng"), build_features(moved, nh, "norm_eng")
    assert np.abs(a - b).max() < 1e-3
    raw_a, raw_b = build_features(coords, nh, "sorted_raw"), build_features(moved, nh, "sorted_raw")
    assert np.abs(raw_a - raw_b).max() > 0.05      # raw coordinates DO change - that is the problem we fixed


def test_hands_are_ordered_left_to_right():
    coords, nh = batch([2])
    swapped = coords.copy()
    swapped[0, [0, 1]] = swapped[0, [1, 0]]
    ordered = canonical_order(swapped, nh)
    assert ordered[0, 0, 0, 0] < ordered[0, 1, 0, 0]
    assert np.allclose(ordered, coords)


def test_mirror_twice_is_identity_and_flips_x():
    coords, nh = batch([1, 2])
    once = mirror(coords, nh)
    assert np.allclose(mirror(once, nh), coords, atol=1e-6)
    assert np.isclose(once[0, 0, 0, 0], 1 - coords[0, 0, 0, 0])


def test_missing_hand_slot_is_zero():
    coords, nh = batch([1])
    X = build_features(coords, nh, "norm")
    assert np.allclose(X[0, 63:126], 0)             # second-hand block
    assert X[0, 126] == 1                           # n_hands


def test_legacy_mode_uses_label_to_choose_slot():
    coords, nh = batch([1])
    left = build_features(coords, nh, "legacy_raw126", np.array([["Left", ""]], dtype=object))
    right = build_features(coords, nh, "legacy_raw126", np.array([["Right", ""]], dtype=object))
    assert np.count_nonzero(left[0, :63]) > 0 and np.count_nonzero(left[0, 63:]) == 0
    assert np.count_nonzero(right[0, 63:]) > 0 and np.count_nonzero(right[0, :63]) == 0


def test_unknown_mode_raises():
    coords, nh = batch([1])
    with pytest.raises(ValueError):
        build_features(coords, nh, "nope")
