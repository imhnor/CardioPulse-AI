# ECGPredict Application Documentation

## Overview

ECGPredict is a full‑stack web application that provides AI‑powered 12‑lead ECG diagnosis using a ResNet1D model trained on the PTB‑XL dataset. The app supports multiple ECG file formats (WFDB, DICOM, XML, SCP‑ECG) and delivers real‑time predictions with a premium medical‑grade UI.

---

## Quick Start

### Backend
```powershell
# One‑click start (uses the built‑in virtual environment)
.\"ECG - application\start.ps1\"
```
Or manually:
```powershell
cd "ECG - application\backend"
..\..\ecg_env\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

### Frontend
Open a browser to **http://localhost:8000/app** or open the static file directly:
```
ECG - application/frontend/index.html
```
If you open the HTML file without the server, edit `app.js` line 5 to point to the correct API base:
```js
const API_BASE = 'http://localhost:8000';
```
---

## Project Structure
```
ECG - application/
├── start.ps1               ← One‑click startup script
│
├── backend/                ← FastAPI server and inference logic
│   ├── app.py
│   ├── upload_handler.py
│   ├── format_detector.py
│   ├── ecg_extractor.py
│   ├── preprocessing.py
│   ├── inference.py
│   └── requirements.txt
│
└── frontend/               ← UI components
    ├── index.html
    ├── style.css
    └── app.js
```
---

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Server health and model status |
| `GET`  | `/api/model-info` | Model architecture metadata |
| `POST` | `/api/upload` | Upload ECG file(s); returns `session_id` and signal info |
| `POST` | `/api/predict` | Run inference on a session; returns probabilities |
| `POST` | `/api/upload-and-predict` | Combined upload + predict step |

### Upload Request
```
POST /api/upload
Content-Type: multipart/form-data

files: <ecg_file>   # For WFDB: both .hea and .dat files
```

### Upload Response (example)
```json
{
  "success": true,
  "session_id": "uuid-...",
  "format": "WFDB",
  "format_display": "WFDB (PhysioNet)",
  "lead_names": ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"],
  "n_leads": 12,
  "fs": 100.0,
  "n_samples": 1000,
  "duration_s": 10.0,
  "patient_info": {}
}
```

### Predict Request & Response
```
POST /api/predict
Content-Type: multipart/form-data

session_id: <uuid>
```
```json
{
  "success": true,
  "probabilities": {"NORM":0.92,"MI":0.05,"STTC":0.03,"CD":0.02,"HYP":0.01},
  "predictions": {"NORM":true,"MI":false,"STTC":false,"CD":false,"HYP":false},
  "top_diagnosis": "Normal ECG",
  "top_code": "NORM",
  "top_confidence": 0.92,
  "is_normal": true,
  "labels_info": [...],
  "preprocessing": {
    "original_fs":100,
    "target_fs":100,
    "resampled":false,
    "padded":false,
    "trimmed":false,
    "lead_order": ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
  }
}
```
---

## Supported ECG Formats
| Format | Extension(s) | Notes |
|--------|-------------|-------|
| WFDB | `.hea` + `.dat` | Both files required |
| DICOM | `.dcm` | Standard DICOM waveform |
| XML | `.xml` | HL7 aECG, Philips, GE, Schiller |
| SCP‑ECG | `.scp` | EN 1064 binary format |
---

## Model Details
| Property | Value |
|----------|-------|
| Architecture | ResNet1D (4‑stage, ~2 M params) |
| Training Data | PTB‑XL (17,221 records) |
| Input Shape | `(1, 12, 1000)` |
| Sampling Rate | 100 Hz |
| Lead Order | I, II, III, aVR, aVL, aVF, V1‑V6 |
| Normalisation | Per‑lead z‑score |
| Output | 5‑class sigmoid (multi‑label) |
| Classes | NORM, MI, STTC, CD, HYP |
| Expected AUROC | 0.90–0.93 |
---

## Error Handling
The UI displays clear messages for:
- **Unsupported format** – file type not recognised
- **Missing leads** – fewer than 12 canonical leads
- **Invalid sampling rate** – outside 50‑10 000 Hz
- **Corrupted file** – NaN/Inf values or unreadable binary
- **Model not found** – `models/resnet1d.pth` missing
---

## Requirements
Python packages (installed via `start.ps1`):
- `fastapi`, `uvicorn`
- `torch`, `numpy`, `scipy`
- `wfdb`, `pydicom`
- `pandas`
---

## UI Highlights
- **Floating diagnosis banner** with colour‑coded background (green for normal, red for abnormal) and subtle pulse animation.
- **Full‑width ECG subplot view** that mimics clinical paper (12‑lead stacked grid + rhythm strip).
- **Download PNG** button for exporting the plotted ECG.
- **Dynamic format selector** supporting WFDB, CSV, MAT, SCP, XML.
---

## Contributing
Feel free to open issues or pull requests. For local development, use the provided virtual environment (`ecg_env`).

---

*End of Documentation*
