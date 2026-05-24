# ECG Prediction Web Application — Implementation Plan

## Overview
Build a full-stack ECG prediction web application that:
- Accepts multi-format ECG uploads (WFDB, DICOM, XML, SCP-ECG)
- Automatically detects format, extracts 12-lead signals, resamples to 100 Hz
- Runs inference through the trained **ResNet1D** model
- Outputs diagnosis, confidence scores (NORM / MI / STTC / CD / HYP)
- Clean medical dashboard UI

---

## Key Technical Details (from codebase analysis)

| Property | Value |
|---|---|
| Model file | `models/resnet1d.pth` |
| Input shape | `(B, 12, 1000)` — 12 leads × 1000 timesteps |
| Sampling rate | **100 Hz** (10 sec recording) |
| Lead order | I, II, III, aVR, aVL, aVF, V1–V6 |
| Normalization | Per-lead z-score: `(x - mean) / (std + 1e-8)` |
| Output classes | 5 labels: `NORM`, `MI`, `STTC`, `CD`, `HYP` |
| Output activation | Sigmoid (multi-label, each independent) |

---

## Proposed Changes

### Structure: `ECG - application/`

```
ECG - application/
├── backend/
│   ├── app.py                  # FastAPI entry point
│   ├── upload_handler.py       # File reception, temp storage
│   ├── format_detector.py      # Auto-detect XML/DICOM/SCP/WFDB
│   ├── ecg_extractor.py        # Extract 12-lead signals per format
│   ├── preprocessing.py        # Resample, validate leads, normalize
│   ├── inference.py            # Load ResNet1D, run prediction
│   └── requirements.txt
└── frontend/
    ├── index.html              # Single-page app
    ├── style.css               # Medical dashboard theme
    └── app.js                  # Upload logic, API calls, UI updates
```

---

### Backend

#### [NEW] `app.py`
FastAPI server with:
- `POST /upload` — upload ECG file, returns detection + extraction results
- `POST /predict` — runs inference on uploaded file, returns diagnoses
- `GET /health` — server health check
- CORS enabled for local frontend

#### [NEW] `format_detector.py`
Auto-detect format by:
1. File extension (`.hea` → WFDB, `.dcm` → DICOM, `.xml` → XML, `.scp` → SCP)
2. Binary magic bytes fallback (DICOM starts with DICM at offset 128)
3. Returns enum: `WFDB | DICOM | XML | SCP`

#### [NEW] `ecg_extractor.py`
Per-format signal extraction:
- **WFDB**: `wfdb.rdsamp()` → `(N, 12)` ndarray
- **DICOM**: `pydicom.dcmread()` → parse waveform sequence
- **XML**: `xml.etree.ElementTree` → parse HL7 aECG / Philips/GE XML schema
- **SCP-ECG**: binary protocol decode (lead data section)
Returns: `signal (N, 12)`, `fs (Hz)`, `lead_names (list[str])`

#### [NEW] `preprocessing.py`
Pipeline:
1. Validate 12 leads present (I, II, III, aVR, aVL, aVF, V1–V6)
2. Reorder leads to canonical PTB-XL order
3. Resample to 100 Hz via `scipy.signal.resample_poly`
4. Trim/pad to exactly 1000 samples
5. Per-lead z-score normalization
6. Transpose to `(12, 1000)` → torch tensor `(1, 12, 1000)`

#### [NEW] `inference.py`
- Loads `ResNet1D` architecture (copied from `train_resnet1d.py`)
- Loads weights from `models/resnet1d.pth`
- Runs forward pass → sigmoid → probabilities
- Returns dict: `{NORM: 0.92, MI: 0.12, STTC: 0.05, CD: 0.08, HYP: 0.03}`

#### [NEW] `upload_handler.py`
- Manages temp file storage (in-memory or `/tmp`)
- Handles WFDB dual-file uploads (`.hea` + `.dat`)
- Returns session ID for subsequent predict call

---

### Frontend

#### [NEW] `index.html`
Single-page medical dashboard with:
- Upload zone (drag & drop + button)
- Real-time processing status stepper
- Result card with per-class confidence bars

#### [NEW] `style.css`
Medical dashboard theme:
- White/light gray background with blue accent (#1565C0)
- Cards with subtle shadows and rounded corners
- Animated progress indicators
- Responsive grid layout

#### [NEW] `app.js`
- Drag-and-drop file upload with `FormData`
- Step-by-step status display (Uploading → Detecting → Extracting → Predicting)
- Result rendering: diagnosis + confidence bars + Normal/Abnormal badge

---

## Verification Plan

### Manual Verification
1. Start FastAPI server: `uvicorn app:app --reload`
2. Open `frontend/index.html` in browser
3. Upload a WFDB ECG file from PTB-XL dataset
4. Verify format detected, leads validated, prediction returned
5. Verify error messages for bad files

### Automated
- `/health` endpoint returns 200
- `/predict` with mock tensor returns valid JSON
