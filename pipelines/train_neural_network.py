"""
train_neural_network.py
-----------------------
Trains a PyTorch MLP multi-label classifier on ECG tabular features.

Steps:
  1. Runs preprocessing_pipeline (skips if already done)
  2. Loads train / val / test splits
  3. Trains a 3-layer MLP with BCEWithLogitsLoss + early stopping
  4. Evaluates on val + test
  5. Saves model weights to models/neural_network.pth
     and model architecture info to models/neural_network_config.pkl

Usage:
  python pipelines/train_neural_network.py

Note:
  Requires PyTorch — install with:
    pip install torch
"""

import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, hamming_loss, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg
from pipelines.preprocessing_pipeline import run as preprocess

# ------------------------------------------------------------
MODEL_PATH      = os.path.join(cfg.MODELS_DIR, "neural_network.pth")
MODEL_CFG_PATH  = os.path.join(cfg.MODELS_DIR, "neural_network_config.pkl")

EPOCHS          = 100
BATCH_SIZE      = 128
LR              = 1e-3
PATIENCE        = 10       # Early stopping patience
HIDDEN_DIMS     = [64, 128, 64]
# ------------------------------------------------------------


def _check_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] PyTorch is not installed. Run:  pip install torch")
        sys.exit(1)


def load_splits():
    train = pd.read_csv(cfg.TRAIN_CSV)
    val   = pd.read_csv(cfg.VAL_CSV)
    test  = pd.read_csv(cfg.TEST_CSV)
    return train, val, test


def build_model(input_dim: int, output_dim: int, hidden_dims: list):
    import torch.nn as nn

    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def evaluate_nn(model, X_np, y_np, split_name: str, threshold=0.5):
    import torch
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_np, dtype=torch.float32))
        probs  = torch.sigmoid(logits).numpy()
    y_pred = (probs >= threshold).astype(int)
    hl    = hamming_loss(y_np, y_pred)
    macro = f1_score(y_np, y_pred, average="macro", zero_division=0)
    print(f"\n-- {split_name} --")
    print(f"  Hamming Loss : {hl:.4f}")
    print(f"  Macro F1     : {macro:.4f}")
    print(classification_report(y_np, y_pred,
                                target_names=cfg.TARGET_LABELS,
                                zero_division=0))


def main():
    _check_torch()
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -- Preprocessing check ---------------------------------------------------
    preprocess()

    # -- Model exists check ----------------------------------------------------
    if os.path.exists(MODEL_PATH):
        print(f"[OK] Model already exists at {MODEL_PATH}")
        print("   Skipping training. Delete the file to retrain.\n")
        ans = input("   Re-train anyway? [y/N]: ").strip().lower()
        if ans != "y":
            return

    # -- Load data ------------------------------------------------------------
    print("\n[1/3] Loading splits ...")
    train, val, test = load_splits()

    X_train = train[cfg.FEATURE_COLS].values.astype(np.float32)
    y_train = train[cfg.TARGET_LABELS].values.astype(np.float32)
    X_val   = val[cfg.FEATURE_COLS].values.astype(np.float32)
    y_val   = val[cfg.TARGET_LABELS].values.astype(np.float32)
    X_test  = test[cfg.FEATURE_COLS].values.astype(np.float32)
    y_test  = test[cfg.TARGET_LABELS].values.astype(np.float32)

    # -- DataLoaders -----------------------------------------------------------
    train_ds  = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_dl  = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # -- Model ------------------------------------------------------------
    print("[2/3] Training MLP Neural Network ...")
    input_dim  = X_train.shape[1]
    output_dim = len(cfg.TARGET_LABELS)
    model      = build_model(input_dim, output_dim, HIDDEN_DIMS).to(device)

    # Compute class pos_weight for imbalance
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)

    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_cnt  = 0
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(torch.tensor(X_val).to(device)),
                torch.tensor(y_val).to(device)
            ).item()

        scheduler.step(val_loss)
        avg_train = epoch_loss / len(train_ds)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Restore best weights
    model.load_state_dict(best_state)

    # -- Evaluate ------------------------------------------------------------
    print("[3/3] Evaluating ...")
    evaluate_nn(model, X_val,  y_val,  "Validation")
    evaluate_nn(model, X_test, y_test, "Test")

    # -- Save ------------------------------------------------------------
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    model_cfg = {
        "input_dim":   input_dim,
        "output_dim":  output_dim,
        "hidden_dims": HIDDEN_DIMS,
    }
    with open(MODEL_CFG_PATH, "wb") as f:
        pickle.dump(model_cfg, f)

    print(f"\n[OK] Model weights saved -> {MODEL_PATH}")
    print(f"   Model config  saved -> {MODEL_CFG_PATH}\n")


if __name__ == "__main__":
    main()
