"""
app.py
------
FastAPI server for the ECG Prediction Web Application.

Endpoints:
  POST /api/upload   — Upload ECG file(s), returns format detection + signal info
  POST /api/predict  — Run model inference on uploaded session
  GET  /api/health   — Server health check
  GET  /api/model-info — Model metadata

Run with:
  uvicorn app:app --reload --port 8000
"""

import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Ensure backend directory is on path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from upload_handler   import create_session, get_session, cleanup_session
from format_detector  import detect_format, get_format_display_name, ECGFormat
from ecg_extractor    import extract_ecg, CANONICAL_LEADS
from preprocessing    import preprocess, ECGValidationError
from inference        import load_model, predict, TARGET_LABELS, LABEL_INFO


# ── Lifespan: pre-load model on startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the model once at startup to avoid cold-start delay."""
    print("[startup] Loading ResNet1D model ...")
    try:
        load_model()
        print("[startup] Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"[startup] WARNING: {e}")
        print("[startup] Server will start but /predict will fail until model is placed at models/resnet1d.pth")
    yield
    print("[shutdown] Cleaning up ...")


# ── App init ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "ECG Prediction API",
    description = "12-lead ECG multi-label classification using ResNet1D on PTB-XL",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve static frontend files
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/app", StaticFiles(directory=_frontend_dir, html=True), name="frontend")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error(code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=code, content={"success": False, "error": detail})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    model_ok = True
    try:
        load_model()
    except Exception:
        model_ok = False

    return {
        "status":    "ok",
        "model_loaded": model_ok,
        "labels":    TARGET_LABELS,
        "timestamp": time.time(),
    }


@app.get("/api/model-info")
async def model_info():
    """Return static model metadata."""
    return {
        "model":       "ResNet1D",
        "dataset":     "PTB-XL",
        "input_shape": [1, 12, 1000],
        "fs":          100,
        "leads":       CANONICAL_LEADS,
        "classes":     [
            {"code": k, "name": v["name"]} for k, v in LABEL_INFO.items()
        ],
        "expected_auroc": "0.90–0.93",
    }


@app.post("/api/upload")
async def upload_ecg(files: list[UploadFile] = File(...)):
    """
    Upload one or more ECG files.
    WFDB requires both .hea and .dat files.

    Returns:
      session_id      — use this in /api/predict
      format          — detected format string
      format_display  — human-readable format name
      lead_names      — leads found in file
      n_leads         — number of leads
      fs              — sampling frequency
      n_samples       — signal length
      duration_s      — recording duration in seconds
      patient_info    — demographics if available
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Create session
    session = create_session()

    try:
        # Save all uploaded files to session directory
        for uf in files:
            content = await uf.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"File '{uf.filename}' is empty.")
            session.add_file(uf.filename, content)

        # Identify primary file
        primary_path = session.get_primary_file()
        if not primary_path:
            raise HTTPException(status_code=400, detail="Could not determine primary ECG file.")

        # Detect format
        fmt = detect_format(primary_path)
        if fmt == ECGFormat.UNKNOWN:
            cleanup_session(session.session_id)
            raise HTTPException(
                status_code=422,
                detail=(
                    "Unsupported file format. Accepted: WFDB (.hea+.dat), "
                    "DICOM (.dcm), XML (.xml), SCP-ECG (.scp)"
                )
            )

        # Extract signal
        try:
            result = extract_ecg(primary_path, fmt)
        except ImportError as e:
            cleanup_session(session.session_id)
            raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")
        except ValueError as e:
            cleanup_session(session.session_id)
            raise HTTPException(status_code=422, detail=f"ECG extraction failed: {e}")
        except Exception as e:
            cleanup_session(session.session_id)
            raise HTTPException(status_code=422, detail=f"Could not read ECG file: {e}")

        return {
            "success":        True,
            "session_id":     session.session_id,
            "format":         fmt.value,
            "format_display": get_format_display_name(fmt),
            "lead_names":     result.lead_names,
            "n_leads":        len(result.lead_names),
            "fs":             result.fs,
            "n_samples":      result.n_samples,
            "duration_s":     round(result.duration_s, 2),
            "patient_info":   result.patient_info,
            "files_received": [f.filename for f in files],
        }

    except HTTPException:
        raise
    except Exception as e:
        cleanup_session(session.session_id)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {traceback.format_exc()}")


@app.post("/api/predict")
async def predict_ecg(session_id: str = Form(...)):
    """
    Run model inference on a previously uploaded ECG session.

    Returns:
      probabilities   — {label: float} sigmoid scores
      predictions     — {label: bool}  threshold 0.5
      top_diagnosis   — highest-probability label name
      is_normal       — bool
      labels_info     — list sorted by probability for UI rendering
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Please upload the ECG file again.")

    primary_path = session.get_primary_file()
    if not primary_path:
        raise HTTPException(status_code=400, detail="No files in session.")

    try:
        # Re-detect format and extract signal
        fmt    = detect_format(primary_path)
        result = extract_ecg(primary_path, fmt)

        # Preprocess
        try:
            prep = preprocess(result)
        except ECGValidationError as e:
            raise HTTPException(status_code=422, detail=f"ECG validation failed: {e}")

        # Inference
        try:
            prediction = predict(prep.tensor)
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=f"Model file not found: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

        # Build response
        response = {
            "success":        True,
            "session_id":     session_id,
            **prediction,
            "preprocessing": {
                "original_fs":  prep.original_fs,
                "target_fs":    100,
                "resampled":    prep.resampled,
                "padded":       prep.padded,
                "trimmed":      prep.trimmed,
                "lead_order":   prep.lead_order,
            },
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {traceback.format_exc()}")
    finally:
        # Clean up temp files after prediction
        cleanup_session(session_id)


@app.post("/api/upload-and-predict")
async def upload_and_predict(files: list[UploadFile] = File(...)):
    """
    Combined single-step endpoint: upload + detect + extract + predict in one call.
    Useful for simple integrations that don't need the two-step flow.
    """
    # Step 1: Upload
    upload_resp = await upload_ecg(files)
    session_id  = upload_resp["session_id"]

    # Step 2: Predict
    from fastapi import Request
    from starlette.datastructures import FormData

    # Re-use predict endpoint logic directly
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=500, detail="Session lost between upload and predict.")

    primary_path = session.get_primary_file()
    fmt          = detect_format(primary_path)
    extraction   = extract_ecg(primary_path, fmt)

    try:
        prep = preprocess(extraction)
    except ECGValidationError as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=422, detail=f"ECG validation failed: {e}")

    try:
        prediction = predict(prep.tensor)
    except Exception as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    cleanup_session(session_id)

    return {
        "success":       True,
        "upload":        upload_resp,
        "preprocessing": {
            "original_fs": prep.original_fs,
            "resampled":   prep.resampled,
            "padded":      prep.padded,
            "trimmed":     prep.trimmed,
        },
        **prediction,
    }
