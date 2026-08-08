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
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.inference import ModelNotReadyError, predict_video

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
# 200MB: generous for a short clip (the pipeline only ever looks at ~16
# sampled frames plus a 5s rPPG window regardless of total length), while
# still catching an accidental full-length-movie upload before it ties up
# a request for minutes.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024

app = FastAPI(title="Hybrid Deepfake Detector")


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


@app.get("/health")
def health():
    """
    Cheap liveness check that does NOT trigger model loading (unlike
    /predict) -- useful for confirming the server itself is up before
    worrying about whether checkpoints exist yet.
    """
    return {"status": "ok"}


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


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Hybrid Deepfake Detector</title>
<style>
  :root { color-scheme: light; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 640px; margin: 60px auto; padding: 0 20px; color: #1a1a1a;
  }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  p.subtitle { color: #666; margin-top: 0; }
  .dropzone {
    border: 2px dashed #999; border-radius: 12px; padding: 40px;
    text-align: center; cursor: pointer;
    transition: border-color .15s, background .15s;
  }
  .dropzone.dragover { border-color: #4a90d9; background: #f0f7ff; }
  .dropzone p { margin: 4px 0; }
  .hint { font-size: 0.85rem; color: #888; }
  #file-input { display: none; }
  button {
    margin-top: 16px; padding: 10px 22px; border: none; border-radius: 8px;
    background: #1a1a1a; color: white; font-size: 1rem; cursor: pointer;
  }
  button:disabled { background: #aaa; cursor: not-allowed; }
  #status { margin-top: 14px; color: #555; min-height: 1.2em; }
  #result {
    margin-top: 20px; padding: 20px; border-radius: 12px; display: none;
  }
  #result.real { background: #e6f7ec; border: 1px solid #34a853; }
  #result.fake { background: #fdeaea; border: 1px solid #ea4335; }
  #result .label { font-size: 1.2rem; font-weight: 600; }
  #result .prob { color: #555; margin-top: 6px; font-size: 0.95rem; }
</style>
</head>
<body>
  <h1>Hybrid Deepfake Detector</h1>
  <p class="subtitle">Upload a video to check whether it's real or AI-generated / manipulated.</p>

  <div class="dropzone" id="dropzone">
    <p id="dropzone-text">Drag a video here, or click to choose a file</p>
    <p class="hint">.mp4, .avi, .mov, .mkv &middot; up to 200MB</p>
    <input type="file" id="file-input" accept=".mp4,.avi,.mov,.mkv" />
  </div>

  <button id="submit-btn" disabled>Analyze video</button>
  <div id="status"></div>
  <div id="result"></div>

<script>
  const dropzone = document.getElementById('dropzone');
  const dropzoneText = document.getElementById('dropzone-text');
  const fileInput = document.getElementById('file-input');
  const submitBtn = document.getElementById('submit-btn');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  let selectedFile = null;

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length) setFile(fileInput.files[0]);
  });

  function setFile(file) {
    selectedFile = file;
    dropzoneText.textContent = `Selected: ${file.name}`;
    submitBtn.disabled = false;
    resultEl.style.display = 'none';
    statusEl.textContent = '';
  }

  submitBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    submitBtn.disabled = true;
    statusEl.textContent = 'Analyzing... this can take a little while on CPU.';
    resultEl.style.display = 'none';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/predict', { method: 'POST', body: formData });
      const data = await response.json();

      if (!response.ok) {
        statusEl.textContent = `Error: ${data.detail || 'unknown error'}`;
        submitBtn.disabled = false;
        return;
      }

      statusEl.textContent = '';
      resultEl.className = data.prediction;
      resultEl.style.display = 'block';
      const pFake = (data.fake_probability * 100).toFixed(1);
      const pReal = (data.real_probability * 100).toFixed(1);
      resultEl.innerHTML =
        '<div class="label">' +
        (data.prediction === 'fake' ? 'Likely AI-generated / manipulated' : 'Likely authentic') +
        '</div>' +
        '<div class="prob">P(fake) = ' + pFake + '% &middot; P(real) = ' + pReal + '%</div>';
    } catch (err) {
      statusEl.textContent = `Request failed: ${err}`;
    } finally {
      submitBtn.disabled = false;
    }
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
