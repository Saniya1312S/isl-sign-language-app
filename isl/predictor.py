"""Frame -> letter. Combines the hand detector, feature builder and classifier."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import DEFAULT_MIN_CONFIDENCE
from .features import build_features, hands_to_arrays
from .model_io import ModelBundle


@dataclass
class Prediction:
    label: Optional[str]                       # None when nothing reliable was recognised
    confidence: float = 0.0
    n_hands: int = 0
    reason: str = "ok"                         # ok | no_hand | low_confidence
    top_k: List[Tuple[str, float]] = field(default_factory=list)


class SignPredictor:
    def __init__(self, bundle: ModelBundle, min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.bundle = bundle
        self.min_confidence = min_confidence

    def predict_hands(self, hands: Sequence) -> Prediction:
        if not hands:
            return Prediction(None, 0.0, 0, "no_hand")
        coords, n_hands, labels = hands_to_arrays(hands)
        X = build_features(coords, n_hands, self.bundle.feature_mode, labels)
        proba = self.bundle.model.predict_proba(X)[0]
        order = np.argsort(proba)[::-1][:3]
        classes = self.bundle.classes
        top = [(str(classes[i]), float(proba[i])) for i in order]
        best, conf = top[0]
        if conf < self.min_confidence:
            return Prediction(None, conf, int(n_hands[0]), "low_confidence", top)
        return Prediction(best, conf, int(n_hands[0]), "ok", top)
