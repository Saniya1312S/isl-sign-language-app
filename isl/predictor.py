"""
Frame -> letter. Combines the hand detector, feature builder and classifier (ML or DL).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .config import DEFAULT_MIN_CONFIDENCE, MODELS_DIR
from .features import build_features, hands_to_arrays
from .model_io import ModelBundle


@dataclass
class Prediction:
    label: Optional[str]                       # None when nothing reliable was recognized
    confidence: float = 0.0
    n_hands: int = 0
    reason: str = "ok"                         # ok | no_hand | low_confidence | no_model
    top_k: List[Tuple[str, float]] = field(default_factory=list)


class DLPredictor:
    """Predictor using PyTorch Deep Learning ResNet-MLP model."""
    def __init__(self, checkpoint_path: Union[str, Path], min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        import torch
        from .dl_model import ISLLandmarkNet

        self.min_confidence = min_confidence
        self.device = torch.device("cpu")
        chk = torch.load(str(checkpoint_path), map_location=self.device)
        self.classes = chk["classes"]
        self.mean = np.array(chk["scaler_mean"], dtype=np.float32)
        self.scale = np.array(chk["scaler_scale"], dtype=np.float32)

        self.model = ISLLandmarkNet(
            in_features=chk["in_features"],
            num_classes=len(self.classes),
            hidden_dim=chk.get("hidden_dim", 256),
            num_blocks=chk.get("num_blocks", 2)
        ).to(self.device)
        self.model.load_state_dict(chk["model_state_dict"])
        self.model.eval()

    def predict_features(self, X: np.ndarray) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
        import torch
        X_scaled = (X - self.mean) / np.where(self.scale == 0, 1.0, self.scale)
        with torch.no_grad():
            t_in = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            logits = self.model(t_in)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        order = np.argsort(proba)[::-1][:3]
        top = [(str(self.classes[i]), float(proba[i])) for i in order]
        return proba, top


class SignPredictor:
    """Unified Predictor for Indian Sign Language: supports Classical ML and PyTorch DL."""
    def __init__(self, bundle_or_path: Union[ModelBundle, str, Path], min_confidence: float = DEFAULT_MIN_CONFIDENCE):
        self.min_confidence = min_confidence
        self.is_dl = False
        self.dl_predictor = None
        self.bundle = None

        if isinstance(bundle_or_path, (str, Path)) and str(bundle_or_path).endswith(".pth"):
            self.dl_predictor = DLPredictor(bundle_or_path, min_confidence=min_confidence)
            self.is_dl = True
            self.classes = self.dl_predictor.classes
            self.feature_mode = "robust_isl"
        elif isinstance(bundle_or_path, ModelBundle):
            self.bundle = bundle_or_path
            self.classes = self.bundle.classes
            self.feature_mode = self.bundle.feature_mode
        else:
            raise ValueError(f"Unsupported predictor source: {bundle_or_path}")

    def predict_hands(self, hands: Sequence) -> Prediction:
        if not hands:
            return Prediction(None, 0.0, 0, "no_hand")
        coords, n_hands, labels = hands_to_arrays(hands)
        X = build_features(coords, n_hands, self.feature_mode, labels)

        if self.is_dl:
            proba, top = self.dl_predictor.predict_features(X)
        else:
            proba = self.bundle.model.predict_proba(X)[0]
            order = np.argsort(proba)[::-1][:3]
            classes = self.classes
            top = [(str(classes[i]), float(proba[i])) for i in order]

        best, conf = top[0]
        if conf < self.min_confidence:
            return Prediction(None, conf, int(n_hands[0]), "low_confidence", top)
        return Prediction(best, conf, int(n_hands[0]), "ok", top)
