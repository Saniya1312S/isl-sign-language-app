"""
Unit tests for model loading and metadata verification.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "final_isl_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"


def test_final_model_exists_and_loads():
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict")


def test_metadata_schema():
    assert META_PATH.exists(), f"Metadata file missing at {META_PATH}"
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))

    required_keys = ["model_name", "feature_mode", "n_features", "class_labels", "n_classes"]
    for k in required_keys:
        assert k in meta, f"Metadata missing required key: {k}"

    assert meta["n_classes"] == 26
    assert meta["class_labels"] == sorted(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    assert meta["feature_mode"] == "norm_eng"
    assert meta["n_features"] == 171


def test_model_prediction_pipeline():
    model = joblib.load(MODEL_PATH)
    # Norm_eng mode has 171 features
    dummy_input = np.zeros((2, 171), dtype=np.float32)
    preds = model.predict(dummy_input)
    assert len(preds) == 2
    assert all(p in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for p in preds)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(dummy_input)
        assert probs.shape == (2, 26)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-4)
