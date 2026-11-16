"""Loads the trained models once per server process (cached) and exposes which modes are available."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import (ALPHABET_META, ALPHABET_MODEL, HINDI_META, HINDI_MODEL, RESULTS_DIR, WORDS_META, WORDS_MODEL, MODELS_DIR)
from .model_io import ModelBundle, load_bundle
from .predictor import SignPredictor

MODE_LABELS = {"alphabet": "Alphabet (A-Z fingerspelling)", "words": "Words", "hindi": "Hindi letters"}
_FILES = {"alphabet": (ALPHABET_MODEL, ALPHABET_META), "words": (WORDS_MODEL, WORDS_META),
          "hindi": (HINDI_MODEL, HINDI_META)}

LOAD_ERRORS: Dict[str, str] = {}


def is_torch_available() -> bool:
    """Check if PyTorch is installed without importing torch into memory."""
    return importlib.util.find_spec("torch") is not None


def get_dl_model_path() -> Optional[Path]:
    p = MODELS_DIR / "isl_dl_model.pth"
    return p if p.exists() else None


def load_bundles() -> Dict[str, ModelBundle]:
    """Only models that exist on disk and load successfully are returned."""
    global LOAD_ERRORS
    LOAD_ERRORS.clear()
    out = {}
    for mode, (model_path, meta_path) in _FILES.items():
        if not model_path.exists():
            continue
        try:
            b = load_bundle(model_path, meta_path)
            if b is not None:
                out[mode] = b
        except Exception as e:
            sk_ver = "unknown"
            if meta_path.exists():
                try:
                    sk_ver = json.loads(meta_path.read_text(encoding="utf-8")).get("sklearn_version", "unknown")
                except Exception:
                    pass
            LOAD_ERRORS[mode] = (
                f"Failed to load {mode} model from '{model_path.name}': {e}. "
                f"Fix: Train with compatible scikit-learn (saved with {sk_ver}) via "
                f"`python scripts/train_alphabet.py --stage final`."
            )
    return out


def build_predictors(bundles: Dict[str, ModelBundle], min_confidence: float) -> Dict[str, SignPredictor]:
    return {m: SignPredictor(b, min_confidence) for m, b in bundles.items()}


def read_result(name: str):
    """Read results/<name> (JSON) or None when that stage has not been run yet."""
    p = RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
