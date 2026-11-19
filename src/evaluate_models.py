"""
Evaluation utilities and standalone model evaluation runner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .data_loader import filter_valid_hands, load_raw_dataframe, to_arrays
from .feature_extraction import build_features

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def load_final_model(model_path: Path | str | None = None) -> Any:
    """Load the trained final ISL model."""
    p = Path(model_path) if model_path else MODELS_DIR / "final_isl_model.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Model not found at: {p}. Run python -m src.train_models first.")
    return joblib.load(p)


def evaluate_dataframe(model: Any, df: pd.DataFrame, mode: str = "norm_eng") -> Dict[str, float]:
    """Evaluate a trained model on a given DataFrame."""
    valid_df = filter_valid_hands(df)
    arr = to_arrays(valid_df)
    X = build_features(arr["coords"], arr["n_hands"], mode=mode, labels=arr["labels"])
    y_true = arr["y"]
    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(macro_f1), "n_samples": len(y_true)}


if __name__ == "__main__":
    model = load_final_model()
    raw = load_raw_dataframe()
    ind = raw[raw["source"] == "Indian"]
    ext = raw[raw["source"] == "ISL_Dataset_2"]

    ind_res = evaluate_dataframe(model, ind)
    ext_res = evaluate_dataframe(model, ext)
    print(f"Indian Studio Dataset Evaluation: {ind_res['accuracy']:.1%} Acc | {ind_res['macro_f1']:.1%} Macro-F1 (n={ind_res['n_samples']})")
    print(f"External Dataset Evaluation:      {ext_res['accuracy']:.1%} Acc | {ext_res['macro_f1']:.1%} Macro-F1 (n={ext_res['n_samples']})")
