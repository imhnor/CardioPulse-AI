# ECGPredict Application

A web application for AI-powered 12-lead ECG diagnosis using a trained **ResNet1D** model on the PTB-XL dataset.

**https://imhnor-cardiopulse-ai.hf.space/**

---

## Quick Start

### 1. Start the backend server

```powershell
# From repo root — uses the ecg_env virtualenv automatically
.\ECG - application\start.ps1
```

Or manually:

```powershell
cd "ECG - application\backend"
..\..ecg_env\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

### 2. Open the frontend

Open in browser: http://localhost:8000/app

Or open directly: `ECG - application/frontend/index.html`

> If opening the HTML file directly (without the server), edit `app.js` line 5:
> `const API_BASE = 'http://localhost:8000';` — this already works for local use.

---

## Project Structure

```
ECG - application/
├── start.ps1               ← One-click startup script
│
├── backend/
│   ├── app.py              ← FastAPI server (endpoints: /upload, /predict, /health)
│   ├── upload_handler.py   ← Session-based temp file storage
│   ├── format_detector.py  ← Auto-detect WFDB / DICOM / XML / SCP-ECG
│   ├── ecg_extractor.py    ← Per-format signal extraction
│   ├── preprocessing.py    ← Validate → resample → normalize → tensor
│   ├── inference.py        ← ResNet1D definition + model load + predict
│   └── requirements.txt
│
└── frontend/
    ├── index.html          ← Single-page medical dashboard
    ├── style.css           ← Premium medical UI theme
    └── app.js              ← Upload + API calls + result rendering
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/health` | Server health + model status |
| `GET`  | `/api/model-info` | Model architecture metadata |
| `POST` | `/api/upload` | Upload ECG file(s) → returns session_id + signal info |
| `POST` | `/api/predict` | Run inference on session → returns probabilities |
| `POST` | `/api/upload-and-predict` | Combined single-step endpoint |

### Upload Request
```
POST /api/upload
Content-Type: multipart/form-data

files: <ecg_file>            # For WFDB: both .hea and .dat files
```

### Upload Response
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

### Predict Request
```
POST /api/predict
Content-Type: multipart/form-data

session_id: <uuid from upload>
```

### Predict Response
```json
{
  "success": true,
  "probabilities": { "NORM": 0.92, "MI": 0.05, "STTC": 0.03, "CD": 0.02, "HYP": 0.01 },
  "predictions":   { "NORM": true, "MI": false, "STTC": false, "CD": false, "HYP": false },
  "top_diagnosis": "Normal ECG",
  "top_code": "NORM",
  "top_confidence": 0.92,
  "is_normal": true,
  "labels_info": [ ... ],
  "preprocessing": {
    "original_fs": 100,
    "target_fs": 100,
    "resampled": false,
    "padded": false,
    "trimmed": false,
    "lead_order": ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
  }
}
```

---

## Supported ECG Formats

| Format | Extension(s) | Notes |
|--------|-------------|-------|
| WFDB | `.hea` + `.dat` | Upload both files together |
| DICOM | `.dcm` | Standard DICOM waveform |
| XML | `.xml` | HL7 aECG, Philips, GE, Schiller |
| SCP-ECG | `.scp` | EN 1064 binary format |

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | ResNet1D (4-stage, ~2M params) |
| Training Data | PTB-XL (17,221 records) |
| Input Shape | `(1, 12, 1000)` |
| Sampling Rate | 100 Hz |
| Lead Order | I, II, III, aVR, aVL, aVF, V1–V6 |
| Normalisation | Per-lead z-score |
| Output | 5-class sigmoid (multi-label) |
| Classes | NORM, MI, STTC, CD, HYP |
| Expected AUROC | 0.90–0.93 |

---

## Error Handling

The application detects and displays clear error messages for:
- **Unsupported format** — file type not recognised
- **Missing leads** — fewer than 12 canonical leads found
- **Invalid sampling rate** — below 50 Hz or above 10,000 Hz
- **Corrupted file** — NaN/Inf values or unreadable binary
- **Model not found** — `models/resnet1d.pth` missing

---

## Requirements

Python packages (auto-installed by `start.ps1`):
- `fastapi`, `uvicorn`
- `torch`, `numpy`, `scipy`
- `wfdb`, `pydicom`
- `pandas`
