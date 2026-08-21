# DeepGuard AI &middot; Hybrid Multi-Modal Deepfake Detection

An enterprise-grade, multi-modal AI deepfake detection system combining **Semantic Feature Extraction (EfficientNet-B0)**, **Temporal Dynamics (ResNet-18 + BiLSTM)**, **Handcrafted Forensics (SRM, rPPG, LBP, Landmark Motion)**, and **Distribution Matching Hybrid Fusion (RBF SVM)** to accurately detect AI-generated and manipulated videos with **98.38% ROC-AUC**.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-CUDA%20Enabled-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-98.38%25-success.svg)](#benchmark-results)

---

## 🌟 Key Highlights & Benchmark Results

Evaluated on the unseen, multi-dataset held-out test split (**930 videos**, zero identity leakage):

| Metric | Score | Significance |
| :--- | :---: | :--- |
| **ROC-AUC** | **98.38%** | Near-optimal ranking discrimination on unseen identities |
| **Balanced Accuracy** | **91.66%** | High performance balanced across minority (real) & majority (fake) classes |
| **Fake Precision** | **99.62%** | Ultra-low false positive rate (788 out of 791 flagged fakes were true fakes) |
| **Real Recall** | **93.88%** | 46 out of 49 genuine real videos correctly verified |
| **Overall Accuracy** | **89.68%** | Robust cross-identity generalization |

### 🔬 Ablation Study

| Model Configuration | Feature Dim | Test Accuracy | Balanced Acc | Macro-F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A: Semantic Only (EfficientNet)** | 257 | 88.39% | 88.09% | 0.6892 | **0.9535** |
| **Model B: Semantic + Temporal (ResNet+LSTM)** | 513 | **91.40%** | **94.50%** | **0.7490** | **0.9863** |
| **Model C: Full Hybrid (+ Forensic + Dist)** | 576 | **89.68%** | **91.66%** | **0.7160** | **0.9838** |

---

## 🏛️ System Architecture

```
                  ┌────────────────────────────────────────┐
                  │              Input Video               │
                  └───────────────────┬────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌──────────────┐              ┌──────────────┐              ┌──────────────┐
│   Semantic   │              │   Temporal   │              │  Forensic    │
│    Branch    │              │    Branch    │              │ Handcrafted  │
│(EfficientNet)│              │(ResNet+LSTM) │              │(SRM+rPPG+LBP)│
└───────┬──────┘              └──────┬───────┘              └──────┬───────┘
        │ (256-D)                    │ (256-D)                     │ (63-D)
        └─────────────────────────────┼─────────────────────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │   Feature Concatenation   │ (575-D)
                        │  + Distribution Matching  │ (+1-D)
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │      RBF Kernel SVM       │ (576-D Input)
                        │    (Multi-Modal Fusion)   │
                        └─────────────┬─────────────┘
                                      ▼
                        ┌───────────────────────────┐
                        │  Prediction: Real vs Fake │
                        │      (ROC-AUC: 0.9838)    │
                        └───────────────────────────┘
```

1. **Semantic Spatial Branch**: EfficientNet-B0 fine-tuned on face crops with mixed-precision AMP to detect spatial boundary blending and facial artifact anomalies.
2. **Temporal Dynamic Branch**: ResNet-18 feature extractor + 2-layer Bidirectional LSTM over 16-frame face sequences tracking inter-frame jitter, blinking rhythm, and motion continuity.
3. **Forensic Handcrafted Branch**:
   - **SRM (Spatial Rich Models)**: High-pass filter residuals in horizontal, vertical, and diagonal directions (24-D).
   - **Texture & LBP Statistics**: Variance, skewness, kurtosis, and Local Binary Pattern histograms (23-D).
   - **Landmark Motion Jitter**: Velocity and acceleration stability of 68 facial landmarks (12-D).
   - **Biological rPPG Signals**: Remote photoplethysmography heart pulse signal variance across face regions (4-D).
4. **Distribution Matching**: Ledoit-Wolf shrinkage Mahalanobis statistical distance scoring against genuine media baseline.
5. **Hybrid Fusion Classifier**: Calibrated RBF-kernel Support Vector Machine (SVM) producing confidence probabilities.

---

## 🌐 Full-Stack Web Application

The repository includes a modern **React + Vite** frontend and **FastAPI** backend.

### Features
- **Modern Dark Glassmorphism UI**: High-end cyber-forensics dashboard with Google Fonts (`Outfit`, `Plus Jakarta Sans`) and smooth animations.
- **Drag-and-Drop Video Uploader**: Live instant video playback preview supporting MP4, AVI, MOV, MKV, WebM.
- **Multi-Branch Telemetry**: Live progress gauges for Semantic, Temporal, Forensic, and Distribution distance signals.
- **JSON Forensic Audit Report Export**: One-click download of full diagnostic audit reports.
- **Universal CORS**: Enabled for seamless integration with external cloud frontends (Vercel, Netlify, Render).

---

## 🚀 Getting Started

### 1. Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/yashsidana/hybrid-deepfake-detector.git
cd hybrid-deepfake-detector

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application

```bash
python start_web.py
```
- Web Application: **`http://127.0.0.1:8000`**
- Interactive API Docs: **`http://127.0.0.1:8000/docs`**

---

## 🔌 API Reference

### `GET /health`
Liveness check with GPU acceleration and device telemetry.
```json
{
  "status": "online",
  "gpu_acceleration": true,
  "device": "NVIDIA Quadro P6000",
  "api_version": "2.0.0"
}
```

### `GET /status`
Reports model checkpoint readiness for the React frontend:
```json
{
  "ready": true,
  "stages": {
    "semantic": true,
    "temporal": true,
    "fusion": true
  },
  "message": null
}
```

### `GET /metrics`
Returns benchmark performance metrics from the held-out test evaluation.

### `POST /predict`
Analyzes an uploaded video file (multipart upload):
```json
{
  "prediction": "fake",
  "verdict": "Manipulated / Deepfake",
  "confidence": 98.4,
  "fake_probability": 0.984,
  "real_probability": 0.016,
  "branch_scores": {
    "semantic_spatial": { "score": 0.952, "label": "Suspicious Spatial Artifacts" },
    "temporal_consistency": { "score": 0.912, "label": "Inter-frame Inconsistency" },
    "distribution_distance": { "mahalanobis_distance": 42.1, "status": "Divergent from genuine distribution" }
  },
  "forensic_signals": {
    "srm_residual_noise": 14.28,
    "texture_anomaly": 11.53,
    "facial_landmark_jitter": 89.2,
    "rppg_biological_pulse": 0.08
  },
  "metadata": {
    "device": "CUDA",
    "frames_sampled": 16,
    "inference_time_ms": 4120.5
  }
}
```

---

## 🛠️ Pipeline CLI Commands

```bash
# 1. Video Sampling & Face Precomputation
python -m src.preprocessing.precompute_faces
python -m src.preprocessing.precompute_temporal
python -m src.preprocessing.precompute_forensic

# 2. Model Training
python -m src.modeling.train_semantic
python -m src.modeling.train_temporal

# 3. Embedding Extraction & Hybrid Fusion
python -m src.modeling.extract_embeddings
python -m src.modeling.train_fusion
python -m src.modeling.test_fusion

# 4. Comparative Ablation Study
python -m src.modeling.run_ablation
```

---

## ☁️ Deployment

### Render Blueprint
This repository includes a `render.yaml` blueprint for one-click deployment on Render:
1. Log in to [Render](https://render.com).
2. Click **New +** &rarr; **Blueprint** &rarr; select this repository.
3. Render automatically builds and deploys the web service with health checks on `/health`.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
