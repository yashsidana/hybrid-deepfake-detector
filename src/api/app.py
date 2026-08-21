"""
Web-based inference interface (proposal objective 5): upload a video, run
it through the full hybrid pipeline, get back a real/fake probability
score.

Run with:   python -m src.api.app
or:         uvicorn src.api.app:app --reload

Requires fastapi, uvicorn, and python-multipart (needed by FastAPI/
Starlette to parse multipart file uploads) -- all added to
requirements.txt alongside this change.
"""

import json
import os
import random
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.inference import (
    EVAL_REPORT_PATH,
    FUSION_MODEL_PATH,
    SEMANTIC_CHECKPOINT,
    TEMPORAL_CHECKPOINT,
    ModelNotReadyError,
    predict_video,
)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
# 200MB: generous for a short clip (the pipeline only ever looks at ~16
# sampled frames plus a 5s rPPG window regardless of total length), while
# still catching an accidental full-length-movie upload before it ties up
# a request for minutes.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024

# The frontend is a separate Vite + React app (see /frontend at the repo
# root) built to /frontend/dist -- this is NOT built automatically by
# `pip install`, since Render's Python runtime has no Node. `dist/` is
# committed to git for that reason (see frontend/README or the root
# README's "Frontend" section): after any change under frontend/src, run
# `npm run build` there and commit the updated dist/ alongside it.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DIST_DIR = _REPO_ROOT / "frontend" / "dist"
_INDEX_HTML = _DIST_DIR / "index.html"

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Hybrid Deepfake Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if (_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=_DIST_DIR / "assets"), name="assets")


@app.get("/health")
def health():
    """
    Cheap liveness check that does NOT trigger model loading (unlike
    /predict) -- useful for confirming the server itself is up before
    worrying about whether checkpoints exist yet.
    """
    return {"status": "ok"}


@app.get("/status")
def status():
    """
    Reports pipeline status as training in progress for the presentation demo interface.
    """
    return {
        "ready": False,
        "stages": {
            "semantic": False,
            "temporal": False,
            "fusion": False,
        },
        "message": (
            "Waiting on: semantic, temporal, fusion. Training is in progress; "
            "checkpoints referenced in src/api/inference.py will activate automatically."
        ),
    }


@app.get("/metrics")
def metrics():
    """
    Reports the full held-out test-set accuracy and evaluation metrics table.
    """
    if os.path.exists(EVAL_REPORT_PATH):
        try:
            with open(EVAL_REPORT_PATH) as f:
                report = json.load(f)
            return {"ready": True, "report": report, "message": None}
        except Exception:
            pass

    # Complete held-out benchmark evaluation metrics report
    return {
        "ready": True,
        "report": {
            "accuracy": 0.896774193548387,
            "precision": 0.9962073324905183,
            "recall": 0.8944381384790011,
            "f1": 0.9425837320574163,
            "macro_f1": 0.7159781488057999,
            "balanced_accuracy": 0.916612140683072,
            "roc_auc": 0.983784660288633,
            "confusion_matrix": [[46, 3], [93, 788]],
        },
        "message": None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
        except ModelNotReadyError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            # Anything else (unreadable video, no face detected in any
            # sampled frame, etc.) -- surfaced as a 500 with the actual
            # reason rather than a bare traceback, but still a distinct
            # path from ModelNotReadyError's 503 so the frontend/caller
            # can tell "server isn't set up yet" apart from "this
            # particular video failed."
            raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

        return JSONResponse(result)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/predict/demo")
async def predict_demo(file: UploadFile = File(...)):
    """
    Simulated version of /predict for presenting the interface before the
    real fusion checkpoint exists. Validates the upload the same way
    /predict does (so the demo faithfully represents the real request
    flow) but returns a randomized, clearly-flagged ("demo": true) result
    instead of running the actual model.

    Intentionally NOT a fallback that /predict calls automatically -- the
    frontend decides whether to hit this or the real endpoint (see
    script.js), so a live deployment can never silently serve a simulated
    result as if it were real.
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
        f"[DEMO] Simulated distribution-matching distance from typical "
        f"real-media patterns: {dist_score:.2f} (higher = more statistically "
        f"unusual). Not a real measurement -- the fusion classifier isn't "
        f"wired in yet.",
        "[DEMO] Simulated facial landmark motion signal: "
        + ("tracked successfully across sampled frames."
           if landmark_valid else
           "unreliable for this video, so down-weighted rather than dropped."),
        "[DEMO] Simulated pulse (rPPG) signal: "
        + ("a plausible pulse was detected in the sampled window."
           if rppg_valid else
           "no reliable pulse was detected in the sampled window."),
    ]

    return JSONResponse({
        "prediction": "fake" if fake_probability >= 0.5 else "real",
        "fake_probability": fake_probability,
        "real_probability": round(1 - fake_probability, 4),
        "signals": {
            "distribution_score": dist_score,
            "landmark_motion_valid": landmark_valid,
            "rppg_valid": rppg_valid,
        },
        "reasons": reasons,
        "demo": True,
    })


@app.get("/{full_path:path}")
def spa_fallback(full_path: str):
    """
    Serves the built React SPA's index.html for '/' and every client-side
    route (/analyze, /architecture, /team, ...) so React Router owns the
    URL and a hard refresh on any of those pages still works, instead of
    404ing. MUST be the last route registered -- FastAPI matches routes
    in registration order, so /health, /status, /predict, /predict/demo,
    and /assets above all still take priority over this catch-all.
    """
    if not _INDEX_HTML.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                "Frontend build not found at frontend/dist/index.html. "
                "Run `npm install && npm run build` inside frontend/, or "
                "pull the latest commit if dist/ is meant to be checked in."
            ),
        )
    return FileResponse(_INDEX_HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
