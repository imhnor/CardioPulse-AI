"""
train_resnet1d.py  --  1D Residual Network (ResNet)
=====================================================

WHY RESNET FOR ECG?
-------------------
Deep plain CNNs suffer from the "vanishing gradient" problem -- gradients
shrink to zero as they propagate back through many layers, making the
network fail to learn anything useful past ~10 layers.

ResNet solves this with SKIP CONNECTIONS (residual blocks):

   x --> [Conv -> BN -> ReLU -> Conv -> BN] --> + --> ReLU
   |                                            ^
   +--------------------------------------------+   (shortcut)

Benefits for ECG:
  1. Deeper model  -> captures BOTH local morphology (QRS) AND long-range
     patterns (RR interval regularity, ST depression duration).
  2. More stable training with class imbalance + pos_weight.
  3. State-of-the-art on PTB-XL in published literature (Ribeiro 2020,
     Hannun 2019, Strodthoff 2021).

Architecture:
  Input: (B, 12, 1000)
  -> Initial Conv(12->64, k=15, stride=2)              [500]
  -> ResBlock(64->64)  x2                              [500]
  -> ResBlock(64->128, stride=2) -> ResBlock(128->128) [250]
  -> ResBlock(128->256, stride=2)-> ResBlock(256->256) [125]
  -> ResBlock(256->512, stride=2)-> ResBlock(512->512) [63]
  -> AdaptiveAvgPool -> Linear(512 -> 5)

  Total params: ~1.8 M (small and fast to train on CPU)

Expected performance (PTB-XL):
  Macro AUROC ~0.90-0.93
  Macro F1    ~0.75-0.82

Usage:
  python pipelines/train_resnet1d.py
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
MODEL_PATH = os.path.join(cfg.MODELS_DIR, "resnet1d.pth")
EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-3
PATIENCE   = 12
# -----------------------------------------------------------------------


# --- Building blocks ---------------------------------------------------

class ResBlock1D(nn.Module):
    """
    Standard 1-D Residual Block.
    If stride > 1 or channels differ, uses a 1x1 conv shortcut.
    Default kernel size is set to 3 to prevent parameter bloat.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1  = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                                padding=pad, bias=False)
        self.bn1    = nn.BatchNorm1d(out_ch)
        self.relu   = nn.ReLU(inplace=True)
        self.conv2  = nn.Conv1d(out_ch, out_ch, kernel, stride=1,
                                padding=pad, bias=False)
        self.bn2    = nn.BatchNorm1d(out_ch)
        self.drop   = nn.Dropout(0.2)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(self.bn2(self.conv2(out)))
        return self.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    """
    Optimized 4-stage 1-D ResNet for 12-lead ECG multi-label classification.
    Refactored to enforce Option A (64 -> 128 -> 256 -> 256 channels) with
    a progressive kernel size bottleneck strategy [7, 5, 3, 3] to align receptive
    fields for cardiac signals.
    Total Params: ~2.0 Million (GPU tensor core friendly).
    """
    def __init__(self, in_channels: int = 12, num_classes: int = 5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )                                                 # -> (64, 500)

        # Stage 1: Capture local wave morphologies (e.g., QRS details)
        self.layer1 = nn.Sequential(
            ResBlock1D(64, 64, kernel=7),
            ResBlock1D(64, 64, kernel=7),
        )                                                 # -> (64, 500)

        # Stage 2: Capture intermediate morphological patterns
        self.layer2 = nn.Sequential(
            ResBlock1D(64,  128, stride=2, kernel=5),
            ResBlock1D(128, 128, kernel=5),
        )                                                 # -> (128, 250)

        # Stage 3: Transition to temporal/rhythmic combinations
        self.layer3 = nn.Sequential(
            ResBlock1D(128, 256, stride=2, kernel=3),
            ResBlock1D(256, 256, kernel=3),
        )                                                 # -> (256, 125)

        # Stage 4: Learn deep global representations without parameter explosion
        self.layer4 = nn.Sequential(
            ResBlock1D(256, 256, stride=2, kernel=3),
            ResBlock1D(256, 256, kernel=3),
        )                                                 # -> (256, ~63)

        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),                              # Increased to 0.5 to mitigate val loss divergence
            nn.Linear(256, num_classes),
        )

    def forward(self, x):                                 # (B, 12, 1000)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.classifier(x)


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
    print(f"Model : ResNet1D  |  Batch: {BATCH_SIZE}  |  Epochs: {EPOCHS}")

    # Total parameters
    model = ResNet1D(in_channels=12, num_classes=len(cfg.TARGET_LABELS))
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

    # --- Train ---
    print(f"\n[Training ResNet1D ...]\n")
    best_val, history = train_model(
        model, train_dl, val_dl, MODEL_PATH,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        pos_weight_vec=pos_weight, device=device,
        weight_decay=1e-3,  # Set stronger regularization to mitigate overfitting/validation divergence
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
