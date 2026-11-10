import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

os.environ["KERAS_HOME"] = str(Path(__file__).resolve().parent.parent / ".keras")
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parent.parent / ".matplotlib")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from isl.data import load_landmarks, to_arrays
from scripts.test_robust import extract_robust_features

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return self.act(out + residual)

class ISLLandmarkNet(nn.Module):
    def __init__(self, in_features, num_classes=26, hidden_dim=256, num_blocks=2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout=0.2) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        h = self.input_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)

def train_dl_model():
    print("Loading landmarks data...", flush=True)
    df = load_landmarks()
    a = to_arrays(df)
    
    print("Extracting robust geometric features...", flush=True)
    X = extract_robust_features(a["coords"], a["n_hands"], a["labels"])
    y = a["y"]
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}", flush=True)
    
    model = ISLLandmarkNet(in_features=X.shape[1], num_classes=len(encoder.classes_), hidden_dim=256, num_blocks=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    
    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, 16):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            correct += (out.argmax(dim=1) == by).sum().item()
            total += len(by)
        scheduler.step()
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                val_correct += (out.argmax(dim=1) == by).sum().item()
                val_total += len(by)
        val_acc = val_correct / val_total
        best_acc = max(best_acc, val_acc)
        print(f"Epoch {epoch:2d}/15 - Loss: {total_loss/total:.4f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}", flush=True)
        
    print(f"Deep Learning Model completed in {time.time() - t0:.1f}s - Best Validation Accuracy: {best_acc:.4f}", flush=True)
    return best_acc

if __name__ == "__main__":
    train_dl_model()
