import os
from enum import Enum

class ECGFormat(str, Enum):
    WFDB = "WFDB"
    CSV = "CSV"
    MAT = "MAT"
    SCP = "SCP"
    XML = "XML"
    UNKNOWN = "UNKNOWN"

def _is_wfdb_header(path: str) -> bool:
    try:
        with open(path, "r", errors="ignore") as f:
            line = f.readline().strip()
        parts = line.split()
        if len(parts) >= 3:
            float(parts[1])
            float(parts[2])
            return True
    except:
        pass
    return False

def detect_format(file_path: str) -> ECGFormat:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".hea", ".dat"):
        return ECGFormat.WFDB
    elif ext == ".csv":
        return ECGFormat.CSV
    elif ext == ".mat":
        return ECGFormat.MAT
    elif ext == ".scp":
        return ECGFormat.SCP
    elif ext == ".xml":
        return ECGFormat.XML

    if _is_wfdb_header(file_path):
        return ECGFormat.WFDB

    return ECGFormat.UNKNOWN

def get_format_display_name(fmt: ECGFormat) -> str:
    return {
        ECGFormat.WFDB: "WFDB (PhysioNet)",
        ECGFormat.CSV: "CSV",
        ECGFormat.MAT: "MATLAB (.mat)",
        ECGFormat.SCP: "SCP-ECG (JSON)",
        ECGFormat.XML: "XML",
        ECGFormat.UNKNOWN: "Unknown",
    }.get(fmt, "Unknown")