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
    lead_names = [_normalise_lead_name(n) for n in meta.sig_name]
    fs = float(meta.fs)
    n  = signal.shape[0]

    return ExtractionResult(
        signal=signal.astype(np.float32),
        fs=fs,
        lead_names=lead_names,
        n_samples=n,
        duration_s=n / fs,
        source_format="WFDB",
    )


# ── DICOM extractor ───────────────────────────────────────────────────────────

def _extract_dicom(dcm_path: str) -> ExtractionResult:
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom package required: pip install pydicom")

    ds = pydicom.dcmread(dcm_path)

    # Patient info
    patient_info = {}
    for tag_name, attr in [("PatientName","PatientName"),("PatientAge","PatientAge"),("PatientSex","PatientSex")]:
        try:
            patient_info[tag_name] = str(getattr(ds, attr))
        except AttributeError:
            pass

    # Sampling frequency
    try:
        fs = float(ds.WaveformSequence[0].SamplingFrequency)
    except (AttributeError, IndexError, TypeError):
        fs = 500.0  # DICOM default

    # Extract waveform channels
    waveform_seq = ds.WaveformSequence[0]
    n_channels   = int(waveform_seq.NumberOfWaveformChannels)
    n_samples    = int(waveform_seq.NumberOfWaveformSamples)

    # Channel definitions → lead names
    lead_names = []
    channel_sens = []
    channel_base = []
    for ch_def in waveform_seq.ChannelDefinitionSequence:
        try:
            lead_names.append(_normalise_lead_name(str(ch_def.ChannelSourceSequence[0].CodeMeaning)))
        except Exception:
            lead_names.append(f"Lead{len(lead_names)+1}")
        # Sensitivity: convert from raw integer → mV
        try:
            sens = float(ch_def.ChannelSensitivity)
        except Exception:
            sens = 1.0
        try:
            baseline = float(ch_def.ChannelBaseline)
        except Exception:
            baseline = 0.0
        channel_sens.append(sens)
        channel_base.append(baseline)

    # Raw multiplex waveform data (interleaved channels)
    raw_data = waveform_seq.WaveformData
    dtype_map = {8: np.int8, 16: np.int16, 32: np.int32}
    bits = int(getattr(waveform_seq, "WaveformBitsAllocated", 16))
    dtype = dtype_map.get(bits, np.int16)

    raw = np.frombuffer(raw_data, dtype=dtype)
    raw = raw.reshape(n_samples, n_channels).astype(np.float32)

    # Apply sensitivity (raw → mV)
    for i in range(n_channels):
        raw[:, i] = (raw[:, i] - channel_base[i]) * channel_sens[i] * 1e-3  # μV → mV

    return ExtractionResult(
        signal=raw,
        fs=fs,
        lead_names=lead_names,
        n_samples=n_samples,
        duration_s=n_samples / fs,
        patient_info=patient_info,
        source_format="DICOM",
    )


# ── XML extractor ─────────────────────────────────────────────────────────────

def _extract_xml(xml_path: str) -> ExtractionResult:
    """
    Handles multiple XML ECG dialects:
      - HL7 aECG (annotated ECG)
      - Philips/GE proprietary XML
      - Generic ECG XML with <LeadData> or <waveform> tags
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Strip namespace from all tags for universal matching
    def strip_ns(tag):
        return tag.split("}")[-1] if "}" in tag else tag

    def find_all_stripped(element, tag):
        return [el for el in element.iter() if strip_ns(el.tag) == tag]

    # Try to find sampling frequency
    fs = 500.0
    for tag in ["SampleRate", "sampleRate", "SamplingRate", "samplingFrequency", "SamplingFrequency", "ChannelSampleCountTotal"]:
        els = find_all_stripped(root, tag)
        if els and els[0].text:
            try:
                fs = float(els[0].text)
                break
            except ValueError:
                pass

    # Find lead data blocks — try multiple schemas
    leads_data = {}

    # Schema 1: <LeadData><LeadID>I</LeadID><WaveFormData>...</WaveFormData>
    lead_data_els = find_all_stripped(root, "LeadData")
    for ld in lead_data_els:
        id_el   = next((el for el in ld.iter() if strip_ns(el.tag) in ("LeadID","leadId","ChannelLabel")), None)
        wave_el = next((el for el in ld.iter() if strip_ns(el.tag) in ("WaveFormData","waveFormData","WaveformData","samples","data")), None)
        if id_el is not None and wave_el is not None and wave_el.text:
            lead_name = _normalise_lead_name(id_el.text or "")
            try:
                samples = np.array([float(v) for v in wave_el.text.replace(",", " ").split()], dtype=np.float32)
                leads_data[lead_name] = samples
            except ValueError:
                pass

    # Schema 2: <waveform><lead id="I">...</lead>
    if not leads_data:
        lead_els = find_all_stripped(root, "lead") + find_all_stripped(root, "Lead")
        for le in lead_els:
            lead_id = le.attrib.get("id") or le.attrib.get("name") or ""
            text    = le.text or ""
            if lead_id and text.strip():
                lead_name = _normalise_lead_name(lead_id)
                try:
                    leads_data[lead_name] = np.array([float(v) for v in text.replace(",", " ").split()], dtype=np.float32)
                except ValueError:
                    pass

    # Schema 3: Philips — <sequence><value>...</value> with channel mapping
    if not leads_data:
        seqs = find_all_stripped(root, "sequence")
        for i, seq in enumerate(seqs[:12]):
            val_el = next((el for el in seq.iter() if strip_ns(el.tag) == "value"), None)
            if val_el is not None and val_el.text:
                lead_name = CANONICAL_LEADS[i] if i < 12 else f"Lead{i+1}"
                try:
                    leads_data[lead_name] = np.array([float(v) for v in val_el.text.replace(",", " ").split()], dtype=np.float32)
                except ValueError:
                    pass

    if not leads_data:
        raise ValueError("Could not extract any lead data from XML file. Schema not recognised.")

    # Build (N, 12) matrix — use intersection with canonical leads
    min_len  = min(len(v) for v in leads_data.values())
    signal   = np.zeros((min_len, 12), dtype=np.float32)
    found    = []
    for i, lead in enumerate(CANONICAL_LEADS):
        if lead in leads_data:
            arr = leads_data[lead][:min_len]
            signal[:, i] = arr
            found.append(lead)

    return ExtractionResult(
        signal=signal,
        fs=fs,
        lead_names=list(leads_data.keys()),
        n_samples=min_len,
        duration_s=min_len / fs,
        source_format="XML",
    )


# ── SCP-ECG extractor ─────────────────────────────────────────────────────────

def _extract_scp(scp_path: str) -> ExtractionResult:
    """
    Minimal SCP-ECG (EN1064) binary decoder.
    Decodes Section 6 (rhythm data) to extract lead samples.
    """
    with open(scp_path, "rb") as f:
        data = f.read()

    # Check if it's actually JSON
    if data.strip().startswith(b"{"):
        import json
        try:
            js = json.loads(data)
            fs = float(js.get("sampling_rate", 500.0))
            leads = js.get("leads", [])
            signals = js.get("signals", [])
            
            leads_data = {}
            for i, lead_name in enumerate(leads):
                if i < len(signals):
                    leads_data[_normalise_lead_name(lead_name)] = np.array(signals[i], dtype=np.float32)
            
            if leads_data:
                n_samples = min(len(v) for v in leads_data.values())
                signal    = np.zeros((n_samples, 12), dtype=np.float32)
                for i, lead in enumerate(CANONICAL_LEADS):
                    if lead in leads_data:
                        signal[:, i] = leads_data[lead][:n_samples]

                return ExtractionResult(
                    signal=signal,
                    fs=fs,
                    lead_names=list(leads_data.keys()),
                    n_samples=n_samples,
                    duration_s=n_samples / fs,
                    source_format="SCP-ECG (JSON)",
                )
        except Exception:
            pass

    # SCP-ECG structure:
    # Bytes 0-3 : file CRC (uint16) + file length (uint32) — actually first 6 bytes are section 0 header
    # Then sections: each section starts with [section_id: uint16] [section_crc: uint16] [section_length: uint32]

    # Find file length from offset 2 (after CRC)
    try:
        file_length = struct.unpack_from("<I", data, 2)[0]
    except struct.error:
        file_length = len(data)

    # Parse section pointer table (Section 0)
    # Section 0 starts at offset 6 (after 2-byte CRC + 4-byte length)
    # Each pointer entry: section_id(uint16) + section_index(uint32) + section_length(uint32) = 10 bytes
    section_pointers = {}
    ptr_offset = 6
    try:
        n_pointers_from_table = (struct.unpack_from("<I", data, ptr_offset + 2 + 2)[0] - 6) // 10
    except struct.error:
        n_pointers_from_table = 12

    offset = ptr_offset + 6  # skip section 0 header
    for _ in range(min(n_pointers_from_table, 20)):
        try:
            sec_id  = struct.unpack_from("<H", data, offset)[0]
            sec_idx = struct.unpack_from("<I", data, offset + 2)[0]
            sec_len = struct.unpack_from("<I", data, offset + 6)[0]
            if sec_id < 20 and sec_idx > 0:
                section_pointers[sec_id] = (sec_idx - 1, sec_len)
            offset += 10
        except struct.error:
            break

    # Try to decode Section 3 (lead identification) and Section 6 (rhythm data)
    fs   = 500.0
    leads_data = {}

    # Section 1: patient data / header info (sampling frequency at known offset)
    if 1 in section_pointers:
        s1_off, s1_len = section_pointers[1]
        try:
            fs = float(struct.unpack_from("<H", data, s1_off + 6 + 10)[0])
        except Exception:
            pass

    # Section 6: rhythm data (raw samples, one block per lead)
    if 6 in section_pointers:
        s6_off, s6_len = section_pointers[6]
        # Section 6 header: 6 bytes (crc+len) + section body
        body_off = s6_off + 6
        try:
            amp_factor = struct.unpack_from("<H", data, body_off + 2)[0] / 1000.0  # nV → μV usually
            n_leads    = struct.unpack_from("<B", data, body_off + 4)[0]
            n_samples  = struct.unpack_from("<I", data, body_off + 6)[0]
            samp_off   = body_off + 10 + n_leads * 4  # skip lead offsets

            for i in range(min(n_leads, 12)):
                lead_name = CANONICAL_LEADS[i] if i < 12 else f"Lead{i+1}"
                lead_data = np.frombuffer(
                    data[samp_off + i*n_samples*2 : samp_off + (i+1)*n_samples*2],
                    dtype=np.int16
                ).astype(np.float32) * amp_factor * 1e-6  # → mV
                leads_data[lead_name] = lead_data
        except Exception:
            pass

    if not leads_data:
        raise ValueError(
            "Could not decode SCP-ECG binary data. "
            "File may be corrupted or use an unsupported SCP variant."
        )

    n_samples = min(len(v) for v in leads_data.values())
    signal    = np.zeros((n_samples, 12), dtype=np.float32)
    for i, lead in enumerate(CANONICAL_LEADS):
        if lead in leads_data:
            signal[:, i] = leads_data[lead][:n_samples]

    return ExtractionResult(
        signal=signal,
        fs=fs,
        lead_names=list(leads_data.keys()),
        n_samples=n_samples,
        duration_s=n_samples / fs,
        source_format="SCP-ECG",
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
        ECGFormat.DICOM: _extract_dicom,
        ECGFormat.XML:   _extract_xml,
        ECGFormat.SCP:   _extract_scp,
    }

    if fmt not in dispatchers:
        raise ValueError(f"Unsupported ECG format: {fmt}")

    return dispatchers[fmt](file_path)
