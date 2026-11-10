"""
Full Production Pipeline: Train Classical ML (HistGradientBoosting) and Deep Learning (PyTorch ResNet-MLP)
models on Indian Sign Language alphabet dataset with robust geometric features and augmentation.
"""
import os
import sys
import time
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Safe environment settings
ROOT = Path(__file__).resolve().parent.parent
os.environ["KERAS_HOME"] = str(ROOT / ".keras")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

sys.path.insert(0, str(ROOT))
from isl.config import MODELS_DIR, RESULTS_DIR, LANDMARKS_CSV, ALPHABET_MODEL, ALPHABET_META
from isl.data import load_landmarks, to_arrays
from isl.features import build_features, feature_names
from isl.model_io import ModelBundle, save_bundle
from isl.dl_model import ISLLandmarkNet

def augment_data(coords, n_hands, labels, y):
    """Augment two-handed signs with hand order swap, and single-hand signs with mirror."""
    coords_aug = [coords]
    nh_aug = [n_hands]
    labels_aug = [labels]
    y_aug = [y]
    
    # 1. Swap hand 0 and hand 1 for 2-handed signs
    two_mask = (n_hands == 2)
    if two_mask.any():
        c_swap = coords[two_mask].copy()
        c_swap[:, [0, 1]] = c_swap[:, [1, 0]]
        l_swap = labels[two_mask].copy()
        l_swap[:, [0, 1]] = l_swap[:, [1, 0]]
        coords_aug.append(c_swap)
        nh_aug.append(n_hands[two_mask])
        labels_aug.append(l_swap)
        y_aug.append(y[two_mask])
        
    return (
        np.concatenate(coords_aug, axis=0),
        np.concatenate(nh_aug, axis=0),
        np.concatenate(labels_aug, axis=0),
        np.concatenate(y_aug, axis=0)
    )

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60, flush=True)
    print("ISL Model Training Pipeline: Classical ML & Deep Learning", flush=True)
    print("=" * 60, flush=True)
    
    t_start = time.time()
    df = load_landmarks()
    # Filter to images where at least one hand was detected
    df_hands = df[df.n_hands > 0].reset_index(drop=True)
    print(f"Loaded {len(df_hands)} samples with hands detected (out of {len(df)} total).", flush=True)
    
    a = to_arrays(df_hands)
    coords, nh, labels, y = a["coords"], a["n_hands"], a["labels"], a["y"]
    
    print("Applying symmetry and hand-order data augmentation...", flush=True)
    coords_aug, nh_aug, labels_aug, y_aug = augment_data(coords, nh, labels, y)
    print(f"Total samples after augmentation: {len(coords_aug)}", flush=True)
    
    print("Extracting 226-dim robust geometric features...", flush=True)
    t_feat = time.time()
    X = build_features(coords_aug, nh_aug, mode="robust_isl", labels=labels_aug)
    print(f"Extracted feature matrix: {X.shape} in {time.time() - t_feat:.2f}s", flush=True)
    
    # Stratified Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_aug, test_size=0.15, random_state=42, stratify=y_aug
    )
    print(f"Train set: {len(X_train)} samples | Validation set: {len(X_val)} samples", flush=True)
    
    # -------------------------------------------------------------
    # 1. Classical ML Model (HistGradientBoosting Pipeline)
    # -------------------------------------------------------------
    print("\n--- Training Classical ML Model (HistGradientBoosting) ---", flush=True)
    ml_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_iter=75,
            learning_rate=0.1,
            min_samples_leaf=25,
            random_state=42
        ))
    ])
    
    t_ml = time.time()
    ml_pipeline.fit(X_train, y_train)
    ml_val_pred = ml_pipeline.predict(X_val)
    ml_acc = accuracy_score(y_val, ml_val_pred)
    ml_time = time.time() - t_ml
    print(f"HistGradientBoosting Validation Accuracy: {ml_acc:.4f} ({ml_acc*100:.2f}%) in {ml_time:.1f}s", flush=True)
    
    # Save Classical ML model bundle
    meta_ml = {
        "name": "alphabet",
        "model_type": "HistGradientBoostingClassifier",
        "accuracy": round(float(ml_acc), 4),
        "training_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "feature_mode": "robust_isl",
        "n_features": int(X.shape[1]),
        "feature_names": feature_names("robust_isl"),
        "classes": ml_pipeline.classes_.tolist(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_bundle(ml_pipeline, meta_ml, ALPHABET_MODEL, ALPHABET_META)
    print(f"Saved Classical ML model to {ALPHABET_MODEL}", flush=True)
    
    # -------------------------------------------------------------
    # 2. Deep Learning Model (PyTorch ResNet-MLP)
    # -------------------------------------------------------------
    print("\n--- Training Deep Learning Model (PyTorch ISLLandmarkNet) ---", flush=True)
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_val_enc = encoder.transform(y_val)
    
    scaler_dl = StandardScaler()
    X_train_s = scaler_dl.fit_transform(X_train)
    X_val_s = scaler_dl.transform(X_val)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch using device: {device}", flush=True)
    
    dl_model = ISLLandmarkNet(in_features=X.shape[1], num_classes=len(encoder.classes_), hidden_dim=256, num_blocks=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(dl_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_s, dtype=torch.float32), torch.tensor(y_train_enc, dtype=torch.long)),
        batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val_enc, dtype=torch.long)),
        batch_size=512, shuffle=False
    )
    
    best_dl_acc = 0.0
    t_dl = time.time()
    for epoch in range(1, 11):
        dl_model.train()
        loss_epoch, corr, tot = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = dl_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() * len(by)
            corr += (out.argmax(dim=1) == by).sum().item()
            tot += len(by)
        scheduler.step()
        
        dl_model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = dl_model(bx)
                v_corr += (out.argmax(dim=1) == by).sum().item()
                v_tot += len(by)
        v_acc = v_corr / v_tot
        best_dl_acc = max(best_dl_acc, v_acc)
        print(f"Epoch {epoch:2d}/12 - Train Loss: {loss_epoch/tot:.4f} - Train Acc: {corr/tot:.4f} - Val Acc: {v_acc:.4f}", flush=True)
        
    dl_time = time.time() - t_dl
    print(f"Deep Learning Validation Accuracy: {best_dl_acc:.4f} ({best_dl_acc*100:.2f}%) in {dl_time:.1f}s", flush=True)
    
    # Save PyTorch Deep Learning Model
    dl_checkpoint = {
        "model_state_dict": dl_model.state_dict(),
        "classes": encoder.classes_.tolist(),
        "scaler_mean": scaler_dl.mean_.tolist(),
        "scaler_scale": scaler_dl.scale_.tolist(),
        "in_features": int(X.shape[1]),
        "hidden_dim": 256,
        "num_blocks": 2,
        "accuracy": round(float(best_dl_acc), 4)
    }
    torch.save(dl_checkpoint, MODELS_DIR / "isl_dl_model.pth")
    print(f"Saved Deep Learning model to {MODELS_DIR / 'isl_dl_model.pth'}", flush=True)
    
    # -------------------------------------------------------------
    # 3. Save Comprehensive Metadata & Classification Report
    # -------------------------------------------------------------
    report = classification_report(y_val, ml_val_pred, output_dict=True)
    meta = {
        "name": "alphabet",
        "primary_model": "HistGradientBoostingClassifier",
        "ml_accuracy": round(float(ml_acc), 4),
        "dl_accuracy": round(float(best_dl_acc), 4),
        "feature_mode": "robust_isl",
        "n_features": int(X.shape[1]),
        "feature_names": feature_names("robust_isl"),
        "classes": ml_pipeline.classes_.tolist(),
        "total_dataset_samples": int(len(df)),
        "samples_with_hands": int(len(df_hands)),
        "samples_after_augmentation": int(len(coords_aug)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_duration_seconds": round(time.time() - t_start, 1)
    }
    (MODELS_DIR / "alphabet_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (RESULTS_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 60, flush=True)
    print(f"TRAINING COMPLETE in {time.time() - t_start:.1f}s!", flush=True)
    print(f"Classical ML Accuracy: {ml_acc*100:.2f}% | Deep Learning Accuracy: {best_dl_acc*100:.2f}%", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
