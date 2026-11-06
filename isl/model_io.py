"""
Save / load a trained model TOGETHER with its metadata.

Why: the original project loaded `.pkl` files from hard-coded D:\\ paths and nothing recorded which
features a model expected - so a model could silently be fed the wrong features. Here every model
is a scikit-learn Pipeline (scaler + classifier) saved next to a small JSON file that records the
feature mode, class names, training data size and metrics.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import sklearn


@dataclass
class ModelBundle:
    model: object                     # fitted sklearn Pipeline (has predict_proba)
    meta: Dict = field(default_factory=dict)

    @property
    def feature_mode(self) -> str:
        return self.meta.get("feature_mode", "norm_eng")

    @property
    def classes(self) -> List[str]:
        return list(self.model.classes_)


def save_bundle(model, meta: Dict, model_path: Path, meta_path: Path) -> None:
    model_path, meta_path = Path(model_path), Path(meta_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta.setdefault("classes", [str(c) for c in model.classes_])
    meta["sklearn_version"] = sklearn.__version__
    meta["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    joblib.dump(model, model_path, compress=3)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load_bundle(model_path: Path, meta_path: Path) -> Optional[ModelBundle]:
    """Return None if the model file does not exist (e.g. the optional words model)."""
    model_path, meta_path = Path(model_path), Path(meta_path)
    if not model_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(model_path)
    if any("version" in str(w.message).lower() for w in caught):
        meta["version_warning"] = ("Model was saved with another scikit-learn version. "
                                   "Re-run scripts/train_alphabet.py if predictions look wrong.")
    return ModelBundle(model, meta)
