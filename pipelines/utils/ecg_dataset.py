"""
pipelines/utils/ecg_dataset.py
-------------------------------
PyTorch Dataset for PTB-XL raw ECG signals.

Each sample:
  signal  : Tensor (12, 1000)  -> 12 leads x 1000 time-steps (100 Hz, 10 sec)
  label   : Tensor (5,)        -> multi-hot binary for [NORM, MI, STTC, CD, HYP]

Signal normalisation: per-lead z-score (mean=0, std=1) computed per sample.
This keeps the model input-agnostic to amplitude differences across devices.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as cfg

try:
    import wfdb
    WFDB_OK = True
except ImportError:
    WFDB_OK = False


def _load_signal(filename_lr: str) -> np.ndarray:
    """
    Load a single ECG record from the PTB-XL dataset.
    Returns ndarray of shape (1000, 12).
    """
    if not WFDB_OK:
        raise ImportError("wfdb is required: pip install wfdb")
    path = os.path.join(cfg.DATASET_DIR, filename_lr)
    signal, _ = wfdb.rdsamp(path)
    return signal.astype(np.float32)          # (1000, 12)


def _normalise(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score normalisation. NaN-safe."""
    mean = signal.mean(axis=0, keepdims=True)
    std  = signal.std(axis=0, keepdims=True) + 1e-8
    return (signal - mean) / std              # (1000, 12)


class ECGSignalDataset(Dataset):
    """
    PyTorch Dataset that streams ECG waveforms from disk.

    Args:
        csv_path   : path to train/val/test CSV produced by preprocessing_pipeline
        augment    : if True, applies random amplitude scaling (training only)
        cache      : if True, pre-loads ALL signals into RAM (faster training,
                     ~3 GB RAM for 14 k records)
    """

    def __init__(self, csv_path: str, augment: bool = False, cache: bool = False):
        self.df      = pd.read_csv(csv_path)
        self.augment = augment
        self.cache   = cache
        self._data   = {}

        if cache:
            print(f"[ECGSignalDataset] Caching {len(self.df)} signals into RAM ...")
            for i, row in self.df.iterrows():
                self._data[i] = _normalise(_load_signal(row["filename_lr"]))
            print("[ECGSignalDataset] Cache complete.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.cache:
            sig = self._data[self.df.index[idx]]
        else:
            sig = _normalise(_load_signal(row["filename_lr"]))   # (1000, 12)

        # Augmentation: random amplitude scale in [0.8, 1.2]
        if self.augment:
            scale = np.random.uniform(0.8, 1.2)
            sig   = sig * scale

        # Transpose -> (12, 1000) for Conv1d (channels first)
        sig_t  = torch.tensor(sig.T, dtype=torch.float32)         # (12, 1000)
        label  = torch.tensor(row[cfg.TARGET_LABELS].values.astype(np.float32))
        return sig_t, label
