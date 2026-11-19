"""
Train and evaluate the ISL A-Z Alphabet Classifier.
Produces:
- models/alphabet_model.pkl
- models/final_isl_model.joblib
- models/label_encoder.pkl
- models/model_metadata.json
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "landmarks_alphabet.csv.gz"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train():
    print(f"Loading landmark dataset from {DATA_PATH}...")
    if not DATA_PATH.exists():
        print(f"Dataset file not found at {DATA_PATH}. Using pre-existing model bundle.")
        return

    from isl.data import load_landmarks, to_arrays
    from isl.features import build_features

    df = load_landmarks(DATA_PATH)
    coords, nh, labels, groups, sources = to_arrays(df)
    
    # Extract norm_eng 171-d features
    print("Building standardized 171-dimensional features (norm_eng)...")
    X = build_features(coords, nh, mode="norm_eng", labels=labels)
    y = np.array(labels)
    classes = sorted(list(set(y)))
    print(f"Dataset loaded: {len(X)} samples across {len(classes)} classes (A-Z).")

    # Train SVM Classifier with RBF kernel and probability support
    print("Training Support Vector Classifier (RBF kernel, C=10.0)...")
    model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    model.fit(X, y)

    # Save models
    pkl_path = MODELS_DIR / "alphabet_model.pkl"
    joblib_path = MODELS_DIR / "final_isl_model.joblib"
    enc_path = MODELS_DIR / "label_encoder.pkl"
    meta_path = MODELS_DIR / "model_metadata.json"

    joblib.dump(model, pkl_path)
    joblib.dump(model, joblib_path)
    joblib.dump(classes, enc_path)

    meta = {
        "model_name": "Support Vector Machine",
        "feature_mode": "norm_eng",
        "n_features": int(X.shape[1]),
        "class_labels": classes,
        "n_classes": len(classes),
        "random_state": 42,
        "default_confidence_threshold": 0.55
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Successfully trained and saved model to {joblib_path} and {pkl_path}")


if __name__ == "__main__":
    train()
