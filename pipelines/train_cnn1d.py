"""
train_cnn1d.py  --  1D Convolutional Neural Network
=====================================================

WHY THIS MODEL FOR ECG?
-----------------------
An ECG signal is a 1-D time series (12 leads x 1000 timesteps at 100 Hz).
A 1D-CNN slides small filters (e.g. width 7) over time and learns local
morphological patterns such as:
  - P-wave shape   -> atrial activity
  - QRS complex    -> ventricular depolarisation (sharp peak)
  - ST-segment     -> ischemia / MI marker
  - T-wave shape   -> repolarisation abnormalities (STTC)

Why NOT just use tabular features (age/sex)?
  Demographics carry weak signal for 5 classes.
  The waveform itself contains all diagnostic information.

Architecture:
  Input: (B, 12, 1000)
  -> Conv1d(12->32, k=7) -> BN -> ReLU -> MaxPool(2)   [500]
  -> Conv1d(32->64, k=5) -> BN -> ReLU -> MaxPool(2)   [250]
  -> Conv1d(64->128,k=5) -> BN -> ReLU -> MaxPool(2)   [125]
  -> Conv1d(128->256,k=3)-> BN -> ReLU -> AdaptiveAvgPool(1)
  -> Linear(256 -> 5)

Expected performance (PTB-XL literature):
  Macro AUROC ~0.85-0.90
  Macro F1    ~0.70-0.75

Usage:
  python pipelines/train_cnn1d.py
"""

import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfg
from pipelines.preprocessing_pipeline import run as preprocess
from pipelines.utils.ecg_dataset import ECGSignalDataset
from pipelines.utils.dl_trainer import train_model, evaluate_loader

# -----------------------------------------------------------------------
MODEL_PATH = os.path.join(cfg.MODELS_DIR, "cnn1d.pth")
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-3
PATIENCE   = 10
# -----------------------------------------------------------------------


class CNN1D(nn.Module):
    """Lightweight 1-D CNN for 12-lead ECG multi-label classification."""

    def __init__(self, in_channels: int = 12, num_classes: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),          # -> (32, 500)
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),          # -> (64, 250)
            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),         # -> (128, 125)
            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                                  # -> (256, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):           # x: (B, 12, 1000)
        return self.classifier(self.encoder(x))


def compute_pos_weight(train_csv: str) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    y  = df[cfg.TARGET_LABELS].values.astype(np.float32)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)


def main():
    preprocess()

    if os.path.exists(MODEL_PATH):
        print(f"[OK] Model already exists -> {MODEL_PATH}")
        ans = input("   Re-train? [y/N]: ").strip().lower()
        if ans != "y":
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Model : CNN1D  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")

    # --- Datasets & Loaders ---
    train_ds = ECGSignalDataset(cfg.TRAIN_CSV, augment=True)
    val_ds   = ECGSignalDataset(cfg.VAL_CSV,   augment=False)
    test_ds  = ECGSignalDataset(cfg.TEST_CSV,  augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model ---
    model      = CNN1D(in_channels=12, num_classes=len(cfg.TARGET_LABELS))
    pos_weight = compute_pos_weight(cfg.TRAIN_CSV)

    # --- Train ---
    print(f"\n[Training CNN1D ...]\n")
    best_val, history = train_model(
        model, train_dl, val_dl, MODEL_PATH,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        pos_weight_vec=pos_weight, device=device,
    )

    # --- Test evaluation ---
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_metrics = evaluate_loader(model, test_dl, device)

    print("\n--- TEST SET RESULTS ---")
    print(f"  Hamming Loss : {test_metrics['hamming_loss']:.4f}")
    print(f"  Macro F1     : {test_metrics['macro_f1']:.4f}")
    print(f"  Micro F1     : {test_metrics['micro_f1']:.4f}")
    print(f"  Macro AUROC  : {test_metrics['macro_auroc']:.4f}")
    print("  Per-class F1 :")
    for label, f1 in zip(cfg.TARGET_LABELS, test_metrics["per_class_f1"]):
        print(f"    {label:<5} : {f1:.4f}")


if __name__ == "__main__":
    main()
