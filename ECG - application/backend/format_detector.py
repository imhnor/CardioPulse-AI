"""
format_detector.py
------------------
Automatically identifies the ECG file format from:
  1. File extension
  2. Binary magic bytes (for ambiguous or renamed files)

Supported formats:
  - WFDB   (.hea / .dat pair or .hea alone)
  - DICOM  (.dcm)
  - XML    (.xml)   — covers HL7 aECG, Philips, GE, Schiller schemas
  - SCP-ECG (.scp)
"""

import os
from enum import Enum
from typing import Optional


class ECGFormat(str, Enum):
    WFDB  = "WFDB"
    DICOM = "DICOM"
    XML   = "XML"
    SCP   = "SCP-ECG"
    UNKNOWN = "UNKNOWN"


# ── Magic byte signatures ─────────────────────────────────────────────────────
_DICOM_MAGIC_OFFSET = 128
_DICOM_MAGIC        = b"DICM"
_XML_MAGIC          = b"<?xml"
_SCP_MAGIC_MIN_LEN  = 6   # SCP-ECG section 0 starts at byte 0, no fixed magic

# Known SCP-ECG section CRC marker (first 2 bytes = section ID 0 = 0x0000)
_SCP_SECTION_ID_0   = b"\x00\x00"


def _read_header(path: str, n: int = 140) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _is_dicom(header: bytes) -> bool:
    return len(header) >= 132 and header[128:132] == _DICOM_MAGIC


def _is_xml(header: bytes) -> bool:
    stripped = header.lstrip(b"\xef\xbb\xbf \t\r\n")   # strip BOM + whitespace
    return stripped[:5] == _XML_MAGIC or stripped[:1] == b"<"


def _is_scp(header: bytes, path: str) -> bool:
    """
    SCP-ECG: no universal magic, but:
    - Extension .scp is the strongest indicator
    - Binary file with first bytes = section 0 CRC (4 bytes) + section 0 ID (2 bytes = 0x0000)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".scp":
        return True
    # Heuristic: binary file, starts with 4-byte CRC + 0x0000 section ID
    if len(header) >= 6 and header[4:6] == _SCP_SECTION_ID_0:
        return True
    return False


def _is_wfdb_header(path: str) -> bool:
    """WFDB .hea files are plain text with specific first-line format."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".hea":
        return True
    try:
        with open(path, "r", errors="ignore") as f:
            first_line = f.readline().strip()
        # WFDB header first line: <record_name> <n_signals> <fs> [<n_samples>] ...
        parts = first_line.split()
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].replace(".", "").isdigit():
            return True
    except OSError:
        pass
    return False


def detect_format(file_path: str) -> ECGFormat:
    """
    Detect the ECG format of the given file.

    Args:
        file_path: absolute path to the uploaded file

    Returns:
        ECGFormat enum value
    """
    ext    = os.path.splitext(file_path)[1].lower()
    header = _read_header(file_path)

    # ── Fast extension checks ─────────────────────────────────────────────────
    if ext in (".hea", ".dat"):
        return ECGFormat.WFDB

    if ext == ".dcm":
        return ECGFormat.DICOM

    if ext == ".xml":
        return ECGFormat.XML

    if ext == ".scp":
        return ECGFormat.SCP

    # ── Magic byte fallback ───────────────────────────────────────────────────
    if _is_dicom(header):
        return ECGFormat.DICOM

    if _is_xml(header):
        return ECGFormat.XML

    if _is_scp(header, file_path):
        return ECGFormat.SCP

    if _is_wfdb_header(file_path):
        return ECGFormat.WFDB

    return ECGFormat.UNKNOWN


def get_format_display_name(fmt: ECGFormat) -> str:
    names = {
        ECGFormat.WFDB:    "WFDB (PhysioNet)",
        ECGFormat.DICOM:   "DICOM ECG",
        ECGFormat.XML:     "XML ECG (HL7 aECG / Vendor)",
        ECGFormat.SCP:     "SCP-ECG",
        ECGFormat.UNKNOWN: "Unknown",
    }
    return names.get(fmt, "Unknown")
