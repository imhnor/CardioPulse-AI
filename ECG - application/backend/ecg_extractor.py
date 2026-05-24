"""
ecg_extractor.py
----------------
Extracts 12-lead ECG signals from supported file formats.

Returns a unified ExtractionResult containing:
  signal      : np.ndarray (N, 12) — raw ADC values in mV
  fs          : float              — sampling frequency in Hz
  lead_names  : list[str]          — lead names as found in file
  n_samples   : int                — number of time samples
  duration_s  : float              — recording duration in seconds
  patient_info: dict               — optional demographics if available
"""

import os
import io
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import xml.etree.ElementTree as ET

from format_detector import ECGFormat


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    signal:       np.ndarray            # (N, 12) float32 in mV
    fs:           float                 # sampling frequency Hz
    lead_names:   list                  # list of 12 lead name strings
    n_samples:    int                   # time dimension
    duration_s:   float                 # recording length seconds
    patient_info: dict = field(default_factory=dict)  # optional demographics
    source_format: str = ""


# ── Lead name normalisation map ───────────────────────────────────────────────
# Maps variant spellings → canonical PTB-XL names
_LEAD_ALIASES = {
    # Standard leads
    "lead i":  "I",  "i":  "I",  "leadi":  "I",
    "lead ii": "II", "ii": "II", "leadii": "II",
    "lead iii":"III","iii":"III","leadiii":"III",
    # Augmented
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "-avr":"aVR",
    "augmented vector right":  "aVR",
    "augmented vector left":   "aVL",
    "augmented vector foot":   "aVF",
    # Precordial
    "v1":"V1","v2":"V2","v3":"V3","v4":"V4","v5":"V5","v6":"V6",
    "chest1":"V1","chest2":"V2","chest3":"V3",
    "chest4":"V4","chest5":"V5","chest6":"V6",
}

# Canonical lead order for PTB-XL model
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _normalise_lead_name(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "").replace("-", "")
    return _LEAD_ALIASES.get(key, raw.strip())


# ── WFDB extractor ────────────────────────────────────────────────────────────

def _extract_wfdb(hea_path: str) -> ExtractionResult:
    try:
        import wfdb
    except ImportError:
        raise ImportError("wfdb package required: pip install wfdb")

    # Strip extension — wfdb.rdsamp needs the base record name
    base = os.path.splitext(hea_path)[0]
    signal, meta = wfdb.rdsamp(base)
    # signal: (N, n_leads) in physical units (mV usually)
    lead_names = [_normalise_lead_name(n) for n in meta['sig_name']]
    fs = float(meta['fs'])
    n  = signal.shape[0]

    return ExtractionResult(
        signal=signal.astype(np.float32),
        fs=fs,
        lead_names=lead_names,
        n_samples=n,
        duration_s=n / fs,
        source_format="WFDB",
    )


# ── CSV Extractor ─────────────────────────────────────────────────────────────
def _extract_csv(path: str) -> ExtractionResult:
    import csv
    with open(path, "r") as f:
        reader = csv.reader(f)
        leads_raw = next(reader) # First row is leads
        data = []
        for row in reader:
            data.append([float(x) for x in row])
    
    signal = np.array(data, dtype=np.float32)
    fs = 100.0 # Default if not specified in CSV
    lead_names = [_normalise_lead_name(n) for n in leads_raw]
    n = signal.shape[0]

    return ExtractionResult(
        signal=signal,
        fs=fs,
        lead_names=lead_names,
        n_samples=n,
        duration_s=n / fs,
        source_format="CSV",
    )

# ── MAT Extractor ─────────────────────────────────────────────────────────────
def _extract_mat(path: str) -> ExtractionResult:
    import scipy.io as sio
    mat = sio.loadmat(path)
    signal = mat['signals'].astype(np.float32)
    fs = float(mat['fs'][0][0]) if np.isscalar(mat['fs'][0][0]) else float(mat['fs'].item())
    
    # mat['leads'] usually comes as array of arrays or strings
    # Try to unpack properly
    raw_leads = mat['leads']
    if isinstance(raw_leads, np.ndarray):
        raw_leads = [str(L).strip("[]'") for L in raw_leads]
    elif isinstance(raw_leads, list):
        raw_leads = [str(L) for L in raw_leads]
        
    lead_names = [_normalise_lead_name(n) for n in raw_leads]
    n = signal.shape[0]

    return ExtractionResult(
        signal=signal,
        fs=fs,
        lead_names=lead_names,
        n_samples=n,
        duration_s=n / fs,
        source_format="MAT",
    )

# ── SCP Extractor (JSON-based from conversion) ─────────────────────────────────
def _extract_scp(path: str) -> ExtractionResult:
    import json
    with open(path, "r") as f:
        data = json.load(f)
        
    signal = np.array(data['signals'], dtype=np.float32)
    fs = float(data['sampling_rate'])
    lead_names = [_normalise_lead_name(n) for n in data['leads']]
    n = signal.shape[0]

    return ExtractionResult(
        signal=signal,
        fs=fs,
        lead_names=lead_names,
        n_samples=n,
        duration_s=n / fs,
        source_format="SCP",
    )

# ── XML Extractor ─────────────────────────────────────────────────────────────
def _extract_xml(path: str) -> ExtractionResult:
    tree = ET.parse(path)
    root = tree.getroot()
    
    fs_elem = root.find("SamplingRate")
    fs = float(fs_elem.text) if fs_elem is not None else 100.0
    
    leads_elem = root.find("Leads")
    raw_leads = []
    signals = []
    
    for lead in leads_elem.findall("Lead"):
        raw_leads.append(lead.get("name"))
        # Parse comma separated values
        vals = [float(x) for x in lead.text.split(",")]
        signals.append(vals)
        
    # Signals are currently list of lists (lead, time). We need (time, lead)
    signal = np.array(signals, dtype=np.float32).T
    lead_names = [_normalise_lead_name(n) for n in raw_leads]
    n = signal.shape[0]

    return ExtractionResult(
        signal=signal,
        fs=fs,
        lead_names=lead_names,
        n_samples=n,
        duration_s=n / fs,
        source_format="XML",
    )


# ── Dispatcher ────────────────────────────────────────────────────────────────

def extract_ecg(file_path: str, fmt: ECGFormat) -> ExtractionResult:
    """
    Extract ECG signal from file based on detected format.

    Args:
        file_path : absolute path to the file
        fmt       : ECGFormat enum value from format_detector

    Returns:
        ExtractionResult with raw signal and metadata
    """
    dispatchers = {
        ECGFormat.WFDB:  _extract_wfdb,
        ECGFormat.CSV:   _extract_csv,
        ECGFormat.MAT:   _extract_mat,
        ECGFormat.SCP:   _extract_scp,
        ECGFormat.XML:   _extract_xml,
    }

    if fmt not in dispatchers:
        raise ValueError(f"Unsupported ECG format: {fmt}")

    return dispatchers[fmt](file_path)
