"""Loads the trained models once per server process (cached) and exposes which modes are available."""
from __future__ import annotations

import json
from typing import Dict, Tuple

from .config import (ALPHABET_META, ALPHABET_MODEL, HINDI_META, HINDI_MODEL, RESULTS_DIR, WORDS_META, WORDS_MODEL)
from .model_io import ModelBundle, load_bundle
from .predictor import SignPredictor

MODE_LABELS = {"alphabet": "Alphabet (A-Z fingerspelling)", "words": "Words", "hindi": "Hindi letters"}
_FILES = {"alphabet": (ALPHABET_MODEL, ALPHABET_META), "words": (WORDS_MODEL, WORDS_META),
          "hindi": (HINDI_MODEL, HINDI_META)}


def load_bundles() -> Dict[str, ModelBundle]:
    """Only models that exist on disk are returned (words / hindi are optional)."""
    out = {}
    for mode, (model_path, meta_path) in _FILES.items():
        b = load_bundle(model_path, meta_path)
        if b is not None:
            out[mode] = b
    return out


def build_predictors(bundles: Dict[str, ModelBundle], min_confidence: float) -> Dict[str, SignPredictor]:
    return {m: SignPredictor(b, min_confidence) for m, b in bundles.items()}


def read_result(name: str):
    """Read results/<name> (JSON) or None when that stage has not been run yet."""
    p = RESULTS_DIR / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
