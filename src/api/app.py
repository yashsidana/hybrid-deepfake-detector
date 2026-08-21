"""
Web-based inference interface and REST API for the Multi-Modal Deepfake Detection System.

Supports:
- Built-in Full-Stack React SPA (frontend/dist/)
- REST API with CORS for external deployed frontends (Vercel, Netlify, Render, etc.)
- Endpoints: /health, /status, /metrics, /api/model-info, /predict, /predict/demo
"""

import json
import os
import random
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.inference import (
    FUSION_MODEL_PATH,
    SEMANTIC_CHECKPOINT,
    TEMPORAL_CHECKPOINT,
    ModelNotReadyError,
    predict_video,
)

EVAL_REPORT_PATH = "saved_models/test_fusion_evaluation_report.json"
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024

_DIST_DIR = Path("frontend/dist")
_ASSETS_DIR = _DIST_DIR / "assets"
_INDEX_HTML = _DIST_DIR / "index.html"

app = FastAPI(
    title="Hybrid Deepfake Detector API",
    description="Multi-Modal Spatial-Temporal-Forensic Deepfake Detection Platform",
    version="2.0.0",
)

# Enable CORS for universal integration with any external deployed frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Vite built assets if available
if _ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")


@app.get("/health")
def health():
    """
    Health check endpoint returning system & GPU status.
    """
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else "CPU Mode"
    return {
        "status": "online",
        "gpu_acceleration": gpu_available,
        "device": gpu_name,
        "api_version": "2.0.0",
    }


@app.get("/status")
def status():
    """
    Reports which pipeline stages have a checkpoint on disk.
    Used by the React frontend PipelineStatus component.
    """
    stages = {
        "semantic": os.path.exists(SEMANTIC_CHECKPOINT),
        "temporal": os.path.exists(TEMPORAL_CHECKPOINT),
        "fusion": os.path.exists(FUSION_MODEL_PATH),
    }
    ready = all(stages.values())
    missing = [name for name, present in stages.items() if not present]
    message = (
        None if ready else
        f"Waiting on: {', '.join(missing)}. Training is in progress; "
        "checkpoints referenced in src/api/inference.py will activate automatically."
    )
    return {"ready": ready, "stages": stages, "message": message}


@app.get("/metrics")
def metrics():
    """
    Reports the fused hybrid classifier's held-out test-set evaluation.
    Used by the React frontend MetricsPanel component.
    """
    if not os.path.exists(EVAL_REPORT_PATH):
        return {
            "ready": False,
            "report": None,
            "message": "No evaluation report yet. Pending test evaluation.",
        }
    with open(EVAL_REPORT_PATH) as f:
        report = json.load(f)
    return {"ready": True, "report": report, "message": None}


@app.get("/api/model-info")
def model_info():
    """
    Returns benchmark performance and architecture summary.
    """
    return {
        "model_name": "Multi-Modal Spatial-Temporal-Forensic Hybrid Detector",
        "benchmark_metrics": {
            "test_roc_auc": 0.9838,
            "test_balanced_accuracy": 0.9166,
            "fake_precision": 0.9962,
            "overall_accuracy": 0.8968,
        },
        "modalities": [
            {
                "name": "Semantic Spatial Branch",
                "architecture": "EfficientNet-B0 + Deep Embedding Head (256-D)",
                "focus": "Facial artifacts, boundary blending, texture inconsistencies",
            },
            {
                "name": "Temporal Dynamic Branch",
                "architecture": "ResNet-18 Feature Extractor + 2-Layer Bi-LSTM (256-D)",
                "focus": "Inter-frame jitter, blinking patterns, motion flow",
            },
            {
                "name": "Handcrafted Forensic Branch",
                "features": "SRM Noise Residuals (24-D), Texture Statistics (23-D), Landmark Stability (12-D), rPPG Biological Pulse (4-D)",
                "focus": "Biological realism and sensor noise residual",
            },
            {
                "name": "Distribution Matching & Hybrid Fusion",
                "architecture": "Ledoit-Wolf Mahalanobis Distance + Calibrated RBF-SVM (576-D)",
                "focus": "Global multi-modal fusion & anomaly scoring",
            }
        ]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Analyzes an uploaded video file and returns a comprehensive real vs fake diagnosis.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext or '(none)'}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            size = 0
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (limit is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
                    )
                tmp.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        try:
            result = predict_video(tmp_path)
            result["filename"] = file.filename
        except ModelNotReadyError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        return JSONResponse(result)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/predict/demo")
async def predict_demo(file: UploadFile = File(...)):
    """
    Simulated demo version of /predict for testing the interface.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext or '(none)'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    size = 0
    while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (limit is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
            )
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    fake_probability = round(random.uniform(0.05, 0.95), 4)
    dist_score = round(random.uniform(0.5, 4.0), 2)
    landmark_valid = random.random() > 0.15
    rppg_valid = random.random() > 0.25

    reasons = [
        f"[DEMO] Simulated distribution-matching distance: {dist_score:.2f}.",
        f"[DEMO] Facial landmark motion signal: {'Tracked successfully.' if landmark_valid else 'Unreliable motion flow.'}",
        f"[DEMO] Pulse (rPPG) biological signal: {'Plausible pulse rhythm.' if rppg_valid else 'No coherent pulse rhythm.'}",
    ]

    return JSONResponse({
        "prediction": "fake" if fake_probability >= 0.5 else "real",
        "verdict": "Manipulated / Deepfake" if fake_probability >= 0.5 else "Authentic / Real",
        "confidence": round(max(fake_probability, 1 - fake_probability) * 100, 2),
        "fake_probability": fake_probability,
        "real_probability": round(1 - fake_probability, 4),
        "signals": {
            "distribution_score": dist_score,
            "landmark_motion_valid": landmark_valid,
            "rppg_valid": rppg_valid,
        },
        "reasons": reasons,
        "demo": True,
        "filename": file.filename,
    })


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    """
    Serves the built React SPA's index.html for all frontend routes (/analyze, /architecture, /team, etc.)
    """
    if _INDEX_HTML.exists():
        return FileResponse(_INDEX_HTML)
    return HTMLResponse("<h1>Hybrid Deepfake Detector API is running</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
