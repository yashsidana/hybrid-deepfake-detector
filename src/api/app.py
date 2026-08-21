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

import os
import random
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.inference import (
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

_FRONTEND_DIR = Path(__file__).parent / "frontend"

app = FastAPI(title="Hybrid Deepfake Detector")
app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(_FRONTEND_DIR / "index.html")


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
    Reports which pipeline stages have a checkpoint on disk, WITHOUT
    loading any of them into memory (that only happens lazily, once, on
    the first real /predict call -- see inference._PipelineBundle).

    This is what the frontend polls on page load to decide whether to
    show live-model predictions or fall back to /predict/demo. Once the
    real semantic/temporal/fusion files are dropped into place (no code
    change needed -- see inference.py's SEMANTIC_CHECKPOINT etc.), this
    endpoint flips to ready=true on its own and the frontend hides the
    demo-mode toggle automatically.
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
        "Waiting on: " + ", ".join(missing) + ". Training is in progress; "
        "this panel will flip to ready automatically once the checkpoints "
        "referenced in src/api/inference.py exist on disk -- no code or "
        "frontend changes needed."
    )
    return {"ready": ready, "stages": stages, "message": message}


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
    return JSONResponse({
        "prediction": "fake" if fake_probability >= 0.5 else "real",
        "fake_probability": fake_probability,
        "real_probability": round(1 - fake_probability, 4),
        "demo": True,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
