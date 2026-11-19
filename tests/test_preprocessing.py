"""
Unit tests for src.preprocessing:
- Image handling and color space conversion
- MediaPipe hand detection
- Canonical spatial ordering
- Symmetrical mirror invariance: mirror(mirror(x)) == x
- Invalid image handling
"""
import numpy as np
import pytest

from src.preprocessing import (
    MediaPipeHandDetector,
    canonical_order,
    draw_landmarks_on_image,
    mirror,
)


def test_invalid_image_handling():
    detector = MediaPipeHandDetector(static_image_mode=True)
    # None image
    assert detector.detect_bgr(None) == []
    # Empty array
    assert detector.detect_bgr(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_canonical_order():
    coords = np.zeros((1, 2, 21, 3), dtype=np.float32)
    # Put right hand first (wrist_x = 0.8), left hand second (wrist_x = 0.2)
    coords[0, 0, 0] = [0.8, 0.5, 0.0]
    coords[0, 1, 0] = [0.2, 0.5, 0.0]
    n_hands = np.array([2])

    ordered = canonical_order(coords, n_hands)
    # After ordering, hand 0 must have wrist_x == 0.2 (image-left first)
    assert ordered[0, 0, 0, 0] == pytest.approx(0.2)
    assert ordered[0, 1, 0, 0] == pytest.approx(0.8)


def test_mirror_invariance_identity():
    """Verify that double mirror reflection is an exact identity transformation."""
    coords = np.zeros((2, 2, 21, 3), dtype=np.float32)
    coords[0, 0, 0] = [0.25, 0.4, 0.1]
    coords[0, 1, 0] = [0.75, 0.6, 0.2]
    coords[1, 0, 0] = [0.35, 0.5, 0.0]
    n_hands = np.array([2, 1])
    labels = np.array([["Left", "Right"], ["Right", ""]], dtype=object)

    m1_coords, m1_labels = mirror(coords, n_hands, labels)
    m2_coords, m2_labels = mirror(m1_coords, n_hands, m1_labels)

    np.testing.assert_allclose(m2_coords, coords, atol=1e-5, err_msg="Coordinates failed mirror identity")
    np.testing.assert_array_equal(m2_labels, labels, err_msg="Labels failed mirror identity")


def test_draw_landmarks():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = draw_landmarks_on_image(img, [])
    assert annotated.shape == img.shape
