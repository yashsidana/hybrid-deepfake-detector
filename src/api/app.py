"""
Web-based inference interface and REST API for the Multi-Modal Deepfake Detection System.

Run with:   python -m src.api.app
or:         python start_web.py
"""

import os
import tempfile
import json
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.inference import ModelNotReadyError, predict_video

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024

app = FastAPI(
    title="Hybrid Multi-Modal Deepfake Detector API",
    description="Spatial-Temporal-Forensic Fusion Deepfake Detection Platform",
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


@app.get("/", response_class=HTMLResponse)
def index():
    return _INDEX_HTML


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DeepGuard AI &middot; Multi-Modal Deepfake Detection</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #06090e;
    --surface: #0e131f;
    --surface-glass: rgba(14, 19, 31, 0.75);
    --surface-card: #141b2d;
    --border: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(59, 130, 246, 0.3);
    --text-primary: #f3f4f6;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.4);
    --fake-crimson: #ef4444;
    --fake-glow: rgba(239, 68, 68, 0.35);
    --real-emerald: #10b981;
    --real-glow: rgba(16, 185, 129, 0.35);
    --purple-cyber: #8b5cf6;
    --radius-lg: 18px;
    --radius-md: 12px;
    --radius-sm: 8px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
    position: relative;
    background-image: 
      radial-gradient(circle at 15% 15%, rgba(59, 130, 246, 0.07) 0%, transparent 40%),
      radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.07) 0%, transparent 40%);
  }

  body::before {
    content: "";
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background-size: 40px 40px;
    background-image: linear-gradient(to right, rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                      linear-gradient(to bottom, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
    pointer-events: none;
    z-index: 0;
  }

  .container {
    max-width: 1180px;
    margin: 0 auto;
    padding: 36px 24px 72px;
    position: relative;
    z-index: 1;
  }

  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 40px;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .brand-logo {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, var(--accent), var(--purple-cyber));
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px var(--accent-glow);
  }

  .brand-logo svg { width: 24px; height: 24px; fill: white; }

  .brand-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(to right, #fff, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .brand-badge {
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 20px;
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid var(--border-accent);
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .header-telemetry {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 0.82rem;
    color: var(--text-muted);
  }

  .telemetry-item {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--surface-card);
    padding: 6px 14px;
    border-radius: 20px;
    border: 1px solid var(--border);
  }

  .dot-pulse {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--real-emerald);
    box-shadow: 0 0 10px var(--real-emerald);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
  }

  .hero {
    text-align: center;
    margin-bottom: 40px;
  }

  .hero h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -1px;
    margin-bottom: 14px;
  }

  .hero h1 span {
    background: linear-gradient(135deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .hero p {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 680px;
    margin: 0 auto;
    line-height: 1.6;
  }

  .app-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
  }

  @media (max-width: 920px) {
    .app-grid { grid-template-columns: 1fr; }
    .hero h1 { font-size: 2.2rem; }
  }

  .card {
    background: var(--surface-glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    position: relative;
    overflow: hidden;
  }

  .card-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    color: var(--text-primary);
  }

  .card-title svg { width: 20px; height: 20px; fill: var(--accent); }

  .dropzone {
    border: 2px dashed rgba(59, 130, 246, 0.35);
    border-radius: var(--radius-md);
    background: rgba(14, 19, 31, 0.5);
    padding: 44px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
  }

  .dropzone:hover, .dropzone.dragover {
    border-color: var(--accent);
    background: rgba(59, 130, 246, 0.08);
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(59, 130, 246, 0.15);
  }

  .upload-icon {
    width: 56px;
    height: 56px;
    background: rgba(59, 130, 246, 0.12);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 16px;
    border: 1px solid var(--border-accent);
  }

  .upload-icon svg { width: 28px; height: 28px; stroke: #60a5fa; fill: none; }

  .dropzone-prompt {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 6px;
  }

  .dropzone-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
  }

  #file-input { display: none; }

  .video-preview-wrapper {
    margin-top: 18px;
    display: none;
    border-radius: var(--radius-md);
    overflow: hidden;
    background: #000;
    border: 1px solid var(--border);
    position: relative;
  }

  .video-preview-wrapper video {
    width: 100%;
    max-height: 240px;
    display: block;
    object-fit: contain;
  }

  .video-file-info {
    padding: 10px 14px;
    background: var(--surface-card);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.82rem;
    color: var(--text-secondary);
  }

  .btn-primary {
    width: 100%;
    margin-top: 20px;
    padding: 15px 24px;
    border: none;
    border-radius: var(--radius-md);
    background: linear-gradient(135deg, var(--accent), #2563eb);
    color: white;
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.25s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
  }

  .btn-primary:hover:not(:disabled) {
    background: linear-gradient(135deg, #60a5fa, var(--accent));
    transform: translateY(-2px);
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.5);
  }

  .btn-primary:disabled {
    background: #1f293d;
    color: #4b5563;
    cursor: not-allowed;
    box-shadow: none;
  }

  .scanning-overlay {
    display: none;
    margin-top: 20px;
    padding: 24px;
    background: rgba(14, 19, 31, 0.95);
    border: 1px solid var(--border-accent);
    border-radius: var(--radius-md);
    position: relative;
    overflow: hidden;
  }

  .scanning-bar-track {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
    overflow: hidden;
    margin: 14px 0 16px;
    position: relative;
  }

  .scanning-bar-fill {
    width: 45%;
    height: 100%;
    background: linear-gradient(90deg, transparent, var(--accent), #a855f7, transparent);
    position: absolute;
    animation: scanAnim 1.6s infinite linear;
  }

  @keyframes scanAnim {
    0% { left: -50%; }
    100% { left: 100%; }
  }

  .scanning-steps {
    list-style: none;
    font-size: 0.82rem;
    color: var(--text-muted);
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-family: 'JetBrains Mono', monospace;
  }

  .scanning-step {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .scanning-step.active {
    color: #60a5fa;
  }

  .empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
  }

  .empty-state svg {
    width: 64px;
    height: 64px;
    stroke: rgba(255, 255, 255, 0.1);
    margin-bottom: 16px;
  }

  .results-container {
    display: none;
  }

  .verdict-banner {
    border-radius: var(--radius-md);
    padding: 24px;
    text-align: center;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
  }

  .verdict-banner.fake {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(153, 27, 27, 0.1));
    border: 1px solid rgba(239, 68, 68, 0.4);
    box-shadow: 0 0 30px var(--fake-glow);
  }

  .verdict-banner.real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 95, 70, 0.1));
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 30px var(--real-glow);
  }

  .verdict-tag {
    font-family: 'Outfit', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
  }

  .verdict-banner.fake .verdict-tag { color: var(--fake-crimson); }
  .verdict-banner.real .verdict-tag { color: var(--real-emerald); }

  .verdict-headline {
    font-family: 'Outfit', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
  }

  .verdict-confidence {
    font-size: 0.95rem;
    color: var(--text-secondary);
  }

  .confidence-stat {
    font-weight: 700;
    color: var(--text-primary);
  }

  .breakdown-section {
    margin-bottom: 22px;
  }

  .section-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
  }

  .branch-card {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    margin-bottom: 10px;
  }

  .branch-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .branch-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .branch-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
  }

  .meter-track {
    width: 100%;
    height: 7px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 4px;
    overflow: hidden;
  }

  .meter-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .branch-tag {
    font-size: 0.74rem;
    color: var(--text-muted);
    margin-top: 6px;
  }

  .signals-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 20px;
  }

  .signal-box {
    background: var(--surface-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px;
    font-size: 0.8rem;
  }

  .signal-name {
    color: var(--text-muted);
    margin-bottom: 4px;
  }

  .signal-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: #93c5fd;
  }

  .btn-secondary {
    width: 100%;
    padding: 12px;
    background: transparent;
    border: 1px solid var(--border-accent);
    color: #93c5fd;
    border-radius: var(--radius-sm);
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: background 0.2s ease;
  }

  .btn-secondary:hover {
    background: rgba(59, 130, 246, 0.1);
  }
</style>
</head>
<body>

<div class="container">
  <header>
    <div class="brand">
      <div class="brand-logo">
        <svg viewBox="0 0 24 24"><path d="M12 2L3 7v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-9-5zm0 4a3 3 0 110 6 3 3 0 010-6zm0 14.9c-3.13-.98-5.83-4.14-5.98-8.9.82-.47 2.61-1 5.98-1 3.37 0 5.16.53 5.98 1-.15 4.76-2.85 7.92-5.98 8.9z"/></svg>
      </div>
      <div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="brand-title">DeepGuard AI</span>
          <span class="brand-badge">v2.0 Hybrid</span>
        </div>
      </div>
    </div>

    <div class="header-telemetry">
      <div class="telemetry-item">
        <span class="dot-pulse"></span>
        <span id="gpu-status">GPU Accelerated</span>
      </div>
      <div class="telemetry-item">
        <span>ROC-AUC: <strong>98.38%</strong></span>
      </div>
    </div>
  </header>

  <section class="hero">
    <h1>Multi-Modal <span>Deepfake Forensics</span></h1>
    <p>Upload a video to execute deep spatial-temporal inspection, biological rPPG pulse verification, and noise residual fusion.</p>
  </section>

  <div class="app-grid">
    <div class="card">
      <div class="card-title">
        <svg viewBox="0 0 24 24"><path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/></svg>
        Video Inspection Input
      </div>

      <div class="dropzone" id="dropzone">
        <div class="upload-icon">
          <svg viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/></svg>
        </div>
        <div class="dropzone-prompt" id="dropzone-prompt">Drag & drop video clip or click to browse</div>
        <div class="dropzone-sub">Supports MP4, AVI, MOV, MKV, WebM (up to 200MB)</div>
        <input type="file" id="file-input" accept=".mp4,.avi,.mov,.mkv,.webm" />
      </div>

      <div class="video-preview-wrapper" id="video-preview-wrapper">
        <video id="video-preview" controls></video>
        <div class="video-file-info">
          <span id="file-name-label">sample.mp4</span>
          <span id="file-size-label">12.4 MB</span>
        </div>
      </div>

      <button id="analyze-btn" class="btn-primary" disabled>
        <svg width="20" height="20" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z"/></svg>
        <span>Analyze Video Integrity</span>
      </button>

      <div class="scanning-overlay" id="scanning-overlay">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600;">
          <span style="color: #93c5fd;">Running Multi-Modal Forensic Pipeline...</span>
          <span id="scan-timer" style="color: var(--text-muted);">0.0s</span>
        </div>
        <div class="scanning-bar-track">
          <div class="scanning-bar-fill"></div>
        </div>
        <ul class="scanning-steps">
          <li class="scanning-step active">&bull; Sampling 16 sequential face frames via MTCNN...</li>
          <li class="scanning-step">&bull; Extracting Spatial Artifacts (EfficientNet-B0)...</li>
          <li class="scanning-step">&bull; Computing Spatio-Temporal Dynamics (ResNet+BiLSTM)...</li>
          <li class="scanning-step">&bull; Extracting rPPG Pulse & SRM Noise Residuals...</li>
          <li class="scanning-step">&bull; Evaluating Distribution Matching & RBF-SVM Fusion...</li>
        </ul>
      </div>
    </div>

    <div class="card">
      <div class="card-title">
        <svg viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>
        Diagnostic Intelligence
      </div>

      <div class="empty-state" id="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 8v4m0 4h.01"/></svg>
        <p>Awaiting video upload to perform forensic verification.</p>
      </div>

      <div class="results-container" id="results-container">
        <div class="verdict-banner" id="verdict-banner">
          <div class="verdict-tag" id="verdict-tag">DETECTION VERDICT</div>
          <div class="verdict-headline" id="verdict-headline">Manipulated / Deepfake</div>
          <div class="verdict-confidence">
            Confidence: <span class="confidence-stat" id="confidence-stat">98.4%</span> &middot; Fake Prob: <span class="confidence-stat" id="fake-prob-stat">0.984</span>
          </div>
        </div>

        <div class="breakdown-section">
          <div class="section-label">
            <span>Branch Anomaly Scores</span>
            <span>Probability</span>
          </div>

          <div class="branch-card">
            <div class="branch-header">
              <span class="branch-name">Semantic Spatial Branch (EfficientNet)</span>
              <span class="branch-val" id="sem-score-val">0.00</span>
            </div>
            <div class="meter-track">
              <div class="meter-fill" id="sem-meter" style="width: 0%; background: #3b82f6;"></div>
            </div>
            <div class="branch-tag" id="sem-tag">Checking facial blending & spatial texture</div>
          </div>

          <div class="branch-card">
            <div class="branch-header">
              <span class="branch-name">Temporal Dynamics (ResNet-18 + BiLSTM)</span>
              <span class="branch-val" id="temp-score-val">0.00</span>
            </div>
            <div class="meter-track">
              <div class="meter-fill" id="temp-meter" style="width: 0%; background: #8b5cf6;"></div>
            </div>
            <div class="branch-tag" id="temp-tag">Evaluating multi-frame motion & blink consistency</div>
          </div>

          <div class="branch-card">
            <div class="branch-header">
              <span class="branch-name">Distribution Distance (Mahalanobis)</span>
              <span class="branch-val" id="dist-score-val">0.00</span>
            </div>
            <div class="meter-track">
              <div class="meter-fill" id="dist-meter" style="width: 0%; background: #f59e0b;"></div>
            </div>
            <div class="branch-tag" id="dist-tag">Distance from authentic distribution baseline</div>
          </div>
        </div>

        <div class="breakdown-section">
          <div class="section-label">Handcrafted Forensic Signals</div>
          <div class="signals-grid">
            <div class="signal-box">
              <div class="signal-name">SRM Noise Residual</div>
              <div class="signal-val" id="srm-val">--</div>
            </div>
            <div class="signal-box">
              <div class="signal-name">rPPG Biological Pulse</div>
              <div class="signal-val" id="rppg-val">--</div>
            </div>
            <div class="signal-box">
              <div class="signal-name">Facial Landmark Jitter</div>
              <div class="signal-val" id="jitter-val">--</div>
            </div>
            <div class="signal-box">
              <div class="signal-name">Latency / Device</div>
              <div class="signal-val" id="latency-val">--</div>
            </div>
          </div>
        </div>

        <button class="btn-secondary" id="export-btn">
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
          <span>Export Forensic Audit Report (JSON)</span>
        </button>
      </div>
    </div>
  </div>
</div>

<script>
  let selectedFile = null;
  let latestAnalysisResult = null;

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const previewWrapper = document.getElementById('video-preview-wrapper');
  const videoPreview = document.getElementById('video-preview');
  const fileNameLabel = document.getElementById('file-name-label');
  const fileSizeLabel = document.getElementById('file-size-label');
  const scanningOverlay = document.getElementById('scanning-overlay');
  const emptyState = document.getElementById('empty-state');
  const resultsContainer = document.getElementById('results-container');
  const exportBtn = document.getElementById('export-btn');

  dropzone.addEventListener('click', () => fileInput.click());
  ['dragenter', 'dragover'].forEach(e => {
    dropzone.addEventListener(e, (evt) => {
      evt.preventDefault();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(e => {
    dropzone.addEventListener(e, (evt) => {
      evt.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  function handleFile(file) {
    selectedFile = file;
    fileNameLabel.textContent = file.name;
    fileSizeLabel.textContent = (file.size / (1024 * 1024)).toFixed(1) + ' MB';
    videoPreview.src = URL.createObjectURL(file);
    previewWrapper.style.display = 'block';
    analyzeBtn.disabled = false;
  }

  analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    analyzeBtn.disabled = true;
    scanningOverlay.style.display = 'block';
    emptyState.style.display = 'none';
    resultsContainer.style.display = 'none';

    let startTime = Date.now();
    let timerInterval = setInterval(() => {
      document.getElementById('scan-timer').textContent = ((Date.now() - startTime) / 1000).toFixed(1) + 's';
    }, 100);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Inference error');
      }

      const data = await response.json();
      latestAnalysisResult = data;
      renderResults(data);
    } catch (err) {
      alert('Analysis error: ' + err.message);
      emptyState.style.display = 'block';
    } finally {
      clearInterval(timerInterval);
      scanningOverlay.style.display = 'none';
      analyzeBtn.disabled = false;
    }
  });

  function renderResults(data) {
    const isFake = data.prediction === 'fake';
    const banner = document.getElementById('verdict-banner');
    banner.className = 'verdict-banner ' + (isFake ? 'fake' : 'real');

    document.getElementById('verdict-tag').textContent = isFake ? 'HIGH-RISK MANIPULATION DETECTED' : 'AUTHENTIC MEDIA VERIFIED';
    document.getElementById('verdict-headline').textContent = data.verdict;
    document.getElementById('confidence-stat').textContent = data.confidence + '%';
    document.getElementById('fake-prob-stat').textContent = data.fake_probability;

    const semScore = data.branch_scores.semantic_spatial.score;
    document.getElementById('sem-score-val').textContent = semScore;
    document.getElementById('sem-meter').style.width = (semScore * 100) + '%';
    document.getElementById('sem-meter').style.background = semScore >= 0.5 ? '#ef4444' : '#10b981';
    document.getElementById('sem-tag').textContent = data.branch_scores.semantic_spatial.label;

    const tempScore = data.branch_scores.temporal_consistency.score;
    document.getElementById('temp-score-val').textContent = tempScore;
    document.getElementById('temp-meter').style.width = (tempScore * 100) + '%';
    document.getElementById('temp-meter').style.background = tempScore >= 0.5 ? '#ef4444' : '#10b981';
    document.getElementById('temp-tag').textContent = data.branch_scores.temporal_consistency.label;

    const distDist = data.branch_scores.distribution_distance.mahalanobis_distance;
    document.getElementById('dist-score-val').textContent = distDist;
    const distPct = Math.min(100, (distDist / 50) * 100);
    document.getElementById('dist-meter').style.width = distPct + '%';
    document.getElementById('dist-meter').style.background = distDist > 35 ? '#ef4444' : '#10b981';
    document.getElementById('dist-tag').textContent = data.branch_scores.distribution_distance.status;

    document.getElementById('srm-val').textContent = data.forensic_signals.srm_residual_noise;
    document.getElementById('rppg-val').textContent = data.forensic_signals.rppg_biological_pulse;
    document.getElementById('jitter-val').textContent = data.forensic_signals.facial_landmark_jitter;
    document.getElementById('latency-val').textContent = data.metadata.inference_time_ms + 'ms (' + data.metadata.device + ')';

    resultsContainer.style.display = 'block';
  }

  exportBtn.addEventListener('click', () => {
    if (!latestAnalysisResult) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(latestAnalysisResult, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute('href', dataStr);
    dlAnchor.setAttribute('download', 'deepfake_forensic_report_' + Date.now() + '.json');
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
  });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
