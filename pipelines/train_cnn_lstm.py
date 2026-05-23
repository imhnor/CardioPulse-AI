"""
train_cnn_lstm.py  --  CNN + Bidirectional LSTM Hybrid
=======================================================

WHY CNN + LSTM FOR ECG?
-----------------------
Two complementary problems exist in ECG analysis:

  1. LOCAL morphology (shapes):  P-wave, QRS, T-wave shapes are
     SPATIAL patterns at a fixed time scale.
     --> Best captured by CNN (kernel slides, detects local shapes)

  2. TEMPORAL rhythms (sequences): Are there regular RR intervals?
     Is atrial activity consistent THROUGHOUT the recording?
     --> Best captured by LSTM (maintains memory across timesteps)

The CNN-LSTM hybrid delegates:
  - CNN : learns a compressed feature map per time-step chunk
  - LSTM: reads the feature sequence and models temporal dynamics

This hybrid approach is widely used in clinical ECG papers and often
outperforms pure CNNs on rhythm-based diagnoses (STTC, CD).

Architecture:
  Input: (B, 12, 1000)
  --> CNN feature extractor (3 conv blocks) -> (B, 128, 125)
  --> Transpose                              -> (B, 125, 128)  [seq of 125 timesteps]
  --> BiLSTM(128 hidden, 2 layers)           -> (B, 125, 256)
  --> Take last timestep                     -> (B, 256)
  --> Dropout -> Linear(256 -> 5)

BiLSTM reads the sequence both forward AND backward, giving the
model full context over the entire 10-second recording.

Expected performance (PTB-XL):
  Macro AUROC ~0.87-0.92
  Macro F1    ~0.72-0.78

Usage:
  python pipelines/train_cnn_lstm.py
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
MODEL_PATH = os.path.join(cfg.MODELS_DIR, "cnn_lstm.pth")
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 5e-4
PATIENCE   = 10
# -----------------------------------------------------------------------


class CNNLSTM(nn.Module):
    """
    Hybrid CNN + Bidirectional LSTM for 12-lead ECG.

    CNN extracts local features, BiLSTM captures temporal context.
    """

    def __init__(self, in_channels: int = 12, num_classes: int = 5,
                 lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()

        # ---- CNN Feature Extractor ----
        self.cnn = nn.Sequential(
            # Block 1: local morphology (P, Q, R, S, T peaks)
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),          # -> (32, 500)
            # Block 2: wider patterns (ST segment, QRS complex)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),          # -> (64, 250)
            # Block 3: full-beat patterns
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),         # -> (128, 125)
        )

        # ---- Bidirectional LSTM ----
        # Reads the 125-step feature sequence; hidden*2 because bidirectional
        self.lstm = nn.LSTM(
            input_size  = 128,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = 0.3 if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(lstm_hidden * 2, num_classes),  # *2 for bidirectional
        )

    def forward(self, x):                              # (B, 12, 1000)
        feat = self.cnn(x)                             # (B, 128, 125)
        seq  = feat.permute(0, 2, 1)                   # (B, 125, 128) for LSTM
        out, _ = self.lstm(seq)                        # (B, 125, 256)
        last = out[:, -1, :]                           # (B, 256) -- last timestep
        return self.classifier(last)


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
    print(f"Model : CNN-BiLSTM  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")

    model = CNNLSTM(in_channels=12, num_classes=len(cfg.TARGET_LABELS))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_params:,}")

    # --- Datasets ---
    train_ds = ECGSignalDataset(cfg.TRAIN_CSV, augment=True)
    val_ds   = ECGSignalDataset(cfg.VAL_CSV,   augment=False)
    test_ds  = ECGSignalDataset(cfg.TEST_CSV,  augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    pos_weight = compute_pos_weight(cfg.TRAIN_CSV)

    print(f"\n[Training CNN-BiLSTM ...]\n")
    best_val, history = train_model(
        model, train_dl, val_dl, MODEL_PATH,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        pos_weight_vec=pos_weight, device=device,
    )

    # --- Test ---
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
