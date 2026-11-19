import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from isl.model_io import load_bundle, save_bundle


def test_roundtrip_with_metadata(tmp_path):
    X = np.random.RandomState(0).rand(60, 5)
    y = np.array(["A", "B", "C"] * 20)
    m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=200))]).fit(X, y)
    save_bundle(m, {"feature_mode": "norm_eng", "name": "t"}, tmp_path / "m.joblib", tmp_path / "m.json")
    b = load_bundle(tmp_path / "m.joblib", tmp_path / "m.json")
    assert b.feature_mode == "norm_eng" and b.classes == ["A", "B", "C"]
    assert b.meta["classes"] == ["A", "B", "C"] and "sklearn_version" in b.meta
    assert (b.model.predict(X) == m.predict(X)).all()


def test_missing_optional_model_returns_none(tmp_path):
    assert load_bundle(tmp_path / "nope.joblib", tmp_path / "nope.json") is None
