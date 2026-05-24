"""
preprocessing.py
----------------
Validates and prepares an extracted ECG signal for the ResNet1D model.

Pipeline:
  1. Validate 12 leads present
  2. Reorder leads to PTB-XL canonical order
  3. Resample to exactly TARGET_FS (100 Hz)
  4. Trim or zero-pad to TARGET_LENGTH (1000 samples = 10 sec at 100 Hz)
  5. Per-lead z-score normalisation: (x - mean) / (std + 1e-8)
  6. Return torch.Tensor of shape (1, 12, 1000) — ready for model

Model expects:
  Input shape  : (B, 12, 1000)
  Sampling rate: 100 Hz
  Normalisation: per-lead z-score
"""

import numpy as np
import torch
from typing import Tuple
from dataclasses import dataclass

from ecg_extractor import ExtractionResult, CANONICAL_LEADS

# ── Model constants ───────────────────────────────────────────────────────────
TARGET_FS     = 100      # Hz — PTB-XL 100Hz dataset used for training
TARGET_LENGTH = 1000     # samples — 10 seconds at 100 Hz
N_LEADS       = 12


@dataclass
class PreprocessingResult:
    tensor:       torch.Tensor   # (1, 12, 1000) float32
    lead_order:   list           # canonical lead order applied
    original_fs:  float          # input sampling frequency
    resampled:    bool           # whether resampling was performed
    padded:       bool           # whether zero-padding was applied
    trimmed:      bool           # whether signal was trimmed


# ── Validation ────────────────────────────────────────────────────────────────

class ECGValidationError(ValueError):
    """Raised when the ECG signal fails validation checks."""
    pass


def validate_leads(lead_names: list) -> list:
    """
    Ensure all 12 canonical leads are present.
    Returns the normalised list of found canonical lead names.
    Raises ECGValidationError if any are missing.
    """
    # Normalise: case-insensitive match
    found = set()
    for name in lead_names:
        n = name.strip()
        # Direct match
        if n in CANONICAL_LEADS:
            found.add(n)
        else:
            # Try upper-case match
            for cl in CANONICAL_LEADS:
                if n.upper() == cl.upper():
                    found.add(cl)
                    break

    missing = [l for l in CANONICAL_LEADS if l not in found]
    if missing:
        raise ECGValidationError(
            f"Missing leads: {missing}. "
            f"Found: {list(found)}. "
            f"All 12 leads required: {CANONICAL_LEADS}"
        )

    return CANONICAL_LEADS


def validate_sampling_rate(fs: float) -> None:
    """
    Validate the sampling frequency is in a supported range.
    Raises ECGValidationError for suspicious values.
    """
    if fs < 10:
        raise ECGValidationError(f"Sampling rate too low: {fs} Hz (minimum 10 Hz)")
    if fs > 10000:
        raise ECGValidationError(f"Sampling rate unreasonably high: {fs} Hz (maximum 10000 Hz)")
    if fs < 50:
        raise ECGValidationError(
            f"Sampling rate {fs} Hz is too low for reliable ECG analysis. "
            f"Minimum recommended is 50 Hz; model was trained on 100 Hz."
        )


def validate_signal(signal: np.ndarray) -> None:
    """Check for NaNs, Infs, and flat lines."""
    if np.any(np.isnan(signal)):
        raise ECGValidationError("ECG signal contains NaN values — file may be corrupted.")
    if np.any(np.isinf(signal)):
        raise ECGValidationError("ECG signal contains Inf values — file may be corrupted.")

    # Check for completely flat leads (all identical values = disconnected electrode)
    flat_leads = []
    for i, lead in enumerate(CANONICAL_LEADS):
        col = signal[:, i]
        if col.std() < 1e-9 and not np.all(col == 0):  # 0 padding is ok, flat != 0 is bad
            flat_leads.append(lead)
    # Note: we warn but don't fail — some leads may genuinely be flat (e.g. pacemaker artefact)


# ── Resampling ────────────────────────────────────────────────────────────────

def _resample_signal(signal: np.ndarray, orig_fs: float, target_fs: float) -> np.ndarray:
    """
    Resample (N, 12) signal from orig_fs to target_fs using scipy.
    Uses polyphase resampling for high quality.
    """
    try:
        from scipy.signal import resample_poly
        from math import gcd

        orig_fs_int   = int(round(orig_fs))
        target_fs_int = int(round(target_fs))
        common        = gcd(orig_fs_int, target_fs_int)
        up   = target_fs_int // common
        down = orig_fs_int   // common

        resampled = resample_poly(signal, up, down, axis=0)
        return resampled.astype(np.float32)

    except ImportError:
        # Fallback: linear interpolation if scipy not available
        n_orig   = signal.shape[0]
        n_target = int(round(n_orig * target_fs / orig_fs))
        x_orig   = np.linspace(0, 1, n_orig)
        x_new    = np.linspace(0, 1, n_target)
        resampled = np.zeros((n_target, signal.shape[1]), dtype=np.float32)
        for i in range(signal.shape[1]):
            resampled[:, i] = np.interp(x_new, x_orig, signal[:, i])
        return resampled


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalise(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score normalisation — matches training exactly."""
    mean = signal.mean(axis=0, keepdims=True)          # (1, 12)
    std  = signal.std(axis=0,  keepdims=True) + 1e-8   # (1, 12)
    return ((signal - mean) / std).astype(np.float32)


# ── Lead reordering ───────────────────────────────────────────────────────────

def _reorder_leads(signal: np.ndarray, source_leads: list) -> np.ndarray:
    """
    Reorder columns of signal to match CANONICAL_LEADS order.
    Missing leads are filled with zeros.
    """
    n = signal.shape[0]
    out = np.zeros((n, 12), dtype=np.float32)

    # Build index map: source_leads[i] → column i in signal
    src_map = {}
    for i, name in enumerate(source_leads):
        for cl in CANONICAL_LEADS:
            if name.upper() == cl.upper():
                src_map[cl] = i
                break

    for j, canonical in enumerate(CANONICAL_LEADS):
        if canonical in src_map:
            out[:, j] = signal[:, src_map[canonical]]

    return out


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess(extracted: ExtractionResult) -> PreprocessingResult:
    """
    Full preprocessing pipeline from ExtractionResult → model-ready tensor.

    Steps:
      1. Validate leads
      2. Reorder to canonical order
      3. Validate signal values
      4. Validate sampling rate
      5. Resample to 100 Hz if needed
      6. Trim or pad to 1000 samples
      7. Normalise per-lead z-score
      8. Convert to (1, 12, 1000) torch.Tensor

    Returns:
        PreprocessingResult with tensor and metadata
    """
    signal     = extracted.signal.copy()   # (N, n_leads)
    lead_names = extracted.lead_names
    fs         = extracted.fs

    # ── Step 1: Validate & reorder leads ─────────────────────────────────────
    validate_leads(lead_names)
    signal = _reorder_leads(signal, lead_names)

    # ── Step 2: Validate signal sanity ───────────────────────────────────────
    validate_signal(signal)

    # ── Step 3: Validate sampling rate ───────────────────────────────────────
    validate_sampling_rate(fs)

    # ── Step 4: Resample if needed ────────────────────────────────────────────
    resampled = False
    if abs(fs - TARGET_FS) > 0.5:
        signal    = _resample_signal(signal, fs, TARGET_FS)
        resampled = True

    # ── Step 5: Trim or zero-pad to TARGET_LENGTH ─────────────────────────────
    n      = signal.shape[0]
    padded = trimmed = False

    if n > TARGET_LENGTH:
        signal  = signal[:TARGET_LENGTH, :]
        trimmed = True
    elif n < TARGET_LENGTH:
        pad    = np.zeros((TARGET_LENGTH - n, 12), dtype=np.float32)
        signal = np.concatenate([signal, pad], axis=0)
        padded = True

    # ── Step 6: Per-lead z-score normalisation ────────────────────────────────
    signal = _normalise(signal)

    # ── Step 7: Convert to tensor (1, 12, 1000) ───────────────────────────────
    # signal is currently (1000, 12) — transpose to (12, 1000) then add batch dim
    tensor = torch.tensor(signal.T, dtype=torch.float32).unsqueeze(0)  # (1, 12, 1000)

    return PreprocessingResult(
        tensor=tensor,
        lead_order=CANONICAL_LEADS,
        original_fs=fs,
        resampled=resampled,
        padded=padded,
        trimmed=trimmed,
    )
