"""Personalisation: calibration storage, CSV round-trip and a real accuracy gain on a new domain."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from isl.config import LANDMARKS_CSV
from isl.data import load_landmarks, to_arrays
from isl.features import build_features, mirror
from isl.model_io import ModelBundle
from isl.personalise import CalibrationSet, personalise

needs_data = pytest.mark.skipif(not LANDMARKS_CSV.exists(), reason="landmark table missing")


def test_calibration_set_counts_clear_and_csv_roundtrip():
    cal = CalibrationSet()
    rng = np.random.RandomState(0)
    for letter, n in (("A", 3), ("B", 2)):
        for _ in range(n):
            cal.add(letter, rng.rand(2, 21, 3), 2)
    assert cal.counts() == {"A": 3, "B": 2} and cal.total() == 5
    clone = CalibrationSet()
    assert clone.load_csv(cal.to_csv()) == 5
    assert clone.counts() == cal.counts()
    cal.clear("A")
    assert cal.counts() == {"B": 2}
    cal.clear()
    assert cal.total() == 0


def test_bad_csv_is_rejected():
    with pytest.raises(ValueError):
        CalibrationSet().load_csv(b"a,b\n1,2\n")


def test_personalise_needs_samples():
    bundle = ModelBundle(Pipeline([("s", StandardScaler()), ("clf", LogisticRegression())]), {"feature_mode": "norm_eng"})
    with pytest.raises(ValueError):
        personalise(bundle, np.zeros((2, 171)), np.array(["A", "B"]), CalibrationSet())


@needs_data
def test_a_few_user_samples_improve_accuracy_on_a_new_domain():
    df = load_landmarks()
    df = df[df.n_hands > 0]
    indian = df[df.source == "Indian"].groupby("label").head(80)       # small studio base set
    a = to_arrays(indian)
    X = np.vstack([build_features(a["coords"], a["n_hands"], "norm_eng"),
                   build_features(mirror(a["coords"], a["n_hands"]), a["n_hands"], "norm_eng")])
    y = np.concatenate([a["y"], a["y"]])
    base = ModelBundle(Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300))]).fit(X, y),
                       {"feature_mode": "norm_eng"})

    ext = df[df.source == "ISL_Dataset_2"].sort_values(["label", "frame_idx"])
    cal, test_parts = CalibrationSet(), []
    for label, g in ext.groupby("label"):
        half = len(g) // 2
        ca = to_arrays(g.iloc[:5])                                       # "user records" 5 samples per letter
        for c, n in zip(ca["coords"], ca["n_hands"]):
            cal.add(label, c, n)
        test_parts.append(g.iloc[half:])
    t = to_arrays(__import__("pandas").concat(test_parts))
    Xt = build_features(t["coords"], t["n_hands"], "norm_eng")
    keep = np.isin(t["y"], base.classes)
    before = (base.model.predict(Xt[keep]) == t["y"][keep]).mean()

    personal = personalise(base, X, y, cal)
    after = (personal.model.predict(Xt[keep]) == t["y"][keep]).mean()
    assert personal.meta["personalised"] and personal.meta["user_samples"]
    assert after > before + 0.05, f"personalisation gave {before:.2f} -> {after:.2f}"
