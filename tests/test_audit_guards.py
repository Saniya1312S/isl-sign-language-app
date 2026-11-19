"""
Guard tests ensuring mathematical integrity, honest protocols, and regression prevention.
"""
import importlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pytest

from isl.config import ALPHABET_META, ALPHABET_MODEL, MODELS_DIR, RESULTS_DIR, ROOT
from isl.features import mirror


def test_results_integrity():
    """Verify results/*.json integrity: no duplicates, valid row sums, dynamic importance, real latencies."""
    final_p = RESULTS_DIR / "final_reports.json"
    assert final_p.exists(), "final_reports.json must exist"
    d = json.loads(final_p.read_text(encoding="utf-8"))

    # 1. block_test != external
    assert d["block_test"] != d["external"], "block_test and external must NOT be the same object!"

    # 2. Confusion matrix row sums == per-class support
    cm = np.array(d["block_test"]["confusion_matrix"])
    support = [c["support"] for c in d["block_test"]["per_class"]]
    np.testing.assert_array_equal(cm.sum(axis=1), support, err_msg="CM row sums must equal per-class support")

    # 3. Dynamic feature importance sums to ~1.0 (not a hardcoded static dict)
    imp = d.get("importance_by_group", {})
    assert imp, "importance_by_group must be present"
    assert abs(sum(imp.values()) - 1.0) < 1e-3, f"Importance must sum to 1.0, got {sum(imp.values())}"
    # Verify values are not the old hardcoded floats (0.52, 0.22, 0.14, 0.07, 0.05)
    assert imp.get("hand shape (21 landmarks)") != 0.52, "Feature importance must be dynamically computed"

    # 4. Latency p95 != p50 * 1.35 and detector latency > 1 ms
    lat_p = RESULTS_DIR / "latency.json"
    assert lat_p.exists(), "latency.json must exist"
    lat = json.loads(lat_p.read_text(encoding="utf-8"))
    p50 = lat["total_ms_p50"]
    p95 = lat["total_ms_p95"]
    assert p95 > p50, "p95 must be greater than p50"
    assert abs(p95 - p50 * 1.35) > 1e-2, f"p95 was artificially scaled by 1.35: p50={p50}, p95={p95}"
    assert lat["stages_ms_p50"]["hand detection (MediaPipe)"] > 1.0, "Detector latency must be > 1ms on real images"


def test_honest_protocols():
    """Ensure protocol hierarchy: Random Split (memorized) >= Held-out Blocks >= External."""
    proto_p = RESULTS_DIR / "protocols.json"
    assert proto_p.exists(), "protocols.json must exist"
    p = json.loads(proto_p.read_text(encoding="utf-8"))

    p1 = p["P1_random_split"]["accuracy"]
    p2 = p["P2_block_split"]["accuracy"]
    p3 = p["P3_external"]["accuracy"]

    assert p1 >= p2 >= p3, f"Honest hierarchy violated: P1 ({p1}) >= P2 ({p2}) >= P3 ({p3})"
    # External accuracy should show noticeable domain shift
    assert p3 < 0.60, f"External accuracy on unseen signers ({p3}) unexpectedly high without calibration"


def test_model_selection():
    """Verify that selected model is non-DL, within 0.005 CV Macro-F1 of the best, and loads cleanly."""
    comp_p = RESULTS_DIR / "model_comparison.json"
    assert comp_p.exists(), "model_comparison.json must exist"
    comp = json.loads(comp_p.read_text(encoding="utf-8"))

    selected = comp["selected_model"]
    assert "PyTorch" not in selected and "ResNet" not in selected, "Shipped model must be a classical ML model, not DL"

    # Best CV Macro-F1 among all evaluated models
    best_cv = max(r["cv_macro_f1"] for r in comp["rows"])
    selected_row = next(r for r in comp["rows"] if r["model"] == selected)
    assert (best_cv - selected_row["cv_macro_f1"]) <= 0.005, (
        f"Selected model CV {selected_row['cv_macro_f1']} is more than 0.005 below best CV {best_cv}"
    )

    # Loads cleanly in current environment
    assert ALPHABET_MODEL.exists(), "alphabet_model.joblib must exist"
    bundle = joblib.load(ALPHABET_MODEL)
    assert hasattr(bundle, "predict"), "Loaded bundle must be a valid scikit-learn model"


def test_readme_matches_results():
    """Verify that numbers in README.md match results/*.json exactly."""
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    proto = json.loads((RESULTS_DIR / "protocols.json").read_text(encoding="utf-8"))
    lat = json.loads((RESULTS_DIR / "latency.json").read_text(encoding="utf-8"))

    p1_acc_str = f"{proto['P1_random_split']['accuracy']:.2%}"
    p2_acc_str = f"{proto['P2_block_split']['accuracy']:.2%}"
    p3_acc_str = f"{proto['P3_external']['accuracy']:.2%}"

    assert p1_acc_str in readme_text, f"README missing P1 accuracy {p1_acc_str}"
    assert p2_acc_str in readme_text, f"README missing P2 accuracy {p2_acc_str}"
    assert p3_acc_str in readme_text, f"README missing P3 accuracy {p3_acc_str}"

    total_p50_str = f"{lat['total_ms_p50']:.1f} ms"
    assert total_p50_str in readme_text, f"README missing total p50 latency {total_p50_str}"


def test_mirror_invariance():
    """Verify mirror symmetry: mirror(mirror(coords, n_hands, labels)) == (coords, labels)."""
    coords = np.array([
        [[[0.2, 0.5, 0.0], [0.3, 0.4, 0.0]], [[0.8, 0.5, 0.0], [0.7, 0.4, 0.0]]]
    ], dtype=np.float32)
    # Pad to 21 landmarks
    coords_full = np.zeros((1, 2, 21, 3), dtype=np.float32)
    coords_full[0, 0, 0] = [0.2, 0.5, 0.0]
    coords_full[0, 1, 0] = [0.8, 0.5, 0.0]
    n_hands = np.array([2])
    labels = np.array([["Left", "Right"]], dtype=object)

    m1_coords, m1_labels = mirror(coords_full, n_hands, labels)
    m2_coords, m2_labels = mirror(m1_coords, n_hands, m1_labels)

    np.testing.assert_allclose(m2_coords, coords_full, atol=1e-5, err_msg="Coordinates failed double-mirror identity")
    np.testing.assert_array_equal(m2_labels, labels, err_msg="Labels failed double-mirror identity")


def test_torch_safety():
    """Verify DL model loading uses weights_only=True."""
    dl_path = MODELS_DIR / "isl_dl_model.pth"
    if not dl_path.exists():
        pytest.skip("isl_dl_model.pth not found")

    try:
        import torch
        # Test that loading with weights_only=True succeeds without warning/error
        ckpt = torch.load(dl_path, map_location="cpu", weights_only=True)
        assert isinstance(ckpt, dict), "DL checkpoint must be a state dict"
        assert "state_dict" in ckpt or "model_state_dict" in ckpt, "Checkpoint must contain state dict"
    except ImportError:
        pytest.skip("PyTorch not installed")


def test_live_cam_headless():
    """Feed 60 synthetic frames through live_cam.py in headless mode and assert clean execution."""
    from live_cam import main as live_cam_main

    # Generate 60 blank 640x480 frames
    fake_frames = (np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(60))
    buffer = live_cam_main(fake_camera=fake_frames, show=False, max_frames=60)
    assert buffer is not None, "live_cam.main should return a SpellingBuffer"
    assert isinstance(buffer.text, str), "SpellingBuffer text should be string"


def test_starlette_image_regression():
    """Verify starlette<1 is pinned in requirements.txt to prevent ASGI HTTP 500 bugs."""
    req_text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"starlette\s*<\s*1", req_text), "requirements.txt must pin starlette<1"
