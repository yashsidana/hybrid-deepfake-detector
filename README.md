# Hybrid Deepfake Detector

A modular AI-based Hybrid Deepfake Detection System that combines **Semantic Feature Extraction**, **Temporal Feature Extraction**, and **Feature Fusion** to accurately classify videos as **Real** or **Fake**.

> **Current Status:** Phase 2 (Temporal Feature Extraction) Completed — Feature Fusion next

---

# Project Overview

The objective of this project is to build an efficient and scalable deepfake detection framework that can eventually be deployed as a web application. The system is designed to analyze uploaded videos, extract meaningful visual and temporal information, and predict whether the uploaded content is authentic or manipulated.

The project follows a modular architecture so that every component can be improved independently.

---

# Project Workflow

```
User Uploads Video
        │
        ▼
Video Preprocessing
        │
        ▼
Frame Sampling
        │
        ▼
Face Detection
        │
        ▼
Semantic Feature Extraction
        │
        ▼
Temporal Feature Extraction
        │
        ▼
Feature Fusion (Upcoming)
        │
        ▼
Binary Classification
        │
        ▼
Real / Fake + Confidence Score
```

---

# Current Features

- Dataset management using KaggleHub
- Automatic train/test split generation
- Video frame sampling
- Face detection preprocessing
- Face image caching for faster training
- EfficientNet-B0 based semantic classifier
- Mixed Precision (AMP) Training
- Checkpoint saving
- Model evaluation

---

# Repository Structure

```
hybrid-deepfake-detector/
│
├── config/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── modeling/
│   ├── inference/
│   ├── api/
│   └── tests/
│
├── saved_models/
│
├── notebooks/
│
├── README.md
└── requirements.txt
```

---

# Dataset

Current Dataset:

**Celeb-DF v2**

Dataset is downloaded automatically using KaggleHub.

```python
import kagglehub

path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
```

Dataset Structure

```
data/
└── raw/
    └── celebdf/
        ├── real/
        └── fake/
```

---

# Preprocessing Pipeline

The preprocessing module converts raw videos into training-ready face images.

Pipeline:

```
Celeb-DF Videos
        │
        ▼
setup_dataset.py
        │
        ▼
create_splits.py
        │
        ▼
frame_sampler.py
        │
        ▼
face_detection.py
        │
        ▼
precompute_faces.py
        │
        ▼
semantic_faces/
        │
        ▼
image_loader.py
```

### Preprocessing Modules

### setup_dataset.py

- Downloads Celeb-DF using KaggleHub
- Organizes videos into Real/Fake folders

---

### create_splits.py

Creates

- Train set
- Validation set
- Test set

using an 80-10-10 split.

---

### frame_sampler.py

- Reads videos using OpenCV
- Uniformly samples frames
- Resizes frames to **224×224**
- Converts frames into tensors

---

### face_detection.py

- Detects facial region
- Crops face
- Resizes face to **224×224**
- Removes unnecessary background

---

### precompute_faces.py

One-time preprocessing step.

Converts

```
Video
    ↓
Frame
    ↓
Face Crop
    ↓
JPEG Image
```

Training is performed on precomputed face images instead of videos, significantly reducing training time.

---

### image_loader.py

Uses a custom CSV-driven `Dataset` that reads `data/splits/train.csv`, `val.csv`, and `test.csv` — no random re-splitting, so the same train/val/test boundaries are used everywhere in the pipeline.

Returns

- Train DataLoader
- Validation DataLoader
- Test DataLoader

---

### precompute_temporal.py

One-time preprocessing step for the temporal branch — mirrors `precompute_faces.py`'s caching pattern.

Converts

```
Video
    ↓
16 Sampled Frames
    ↓
Face Crop (per frame)
    ↓
Cached Sequence (.npy, shape [16, 224, 224, 3])
```

Resumable — already-cached videos are skipped on a rerun, so an interrupted preprocessing run can pick up where it left off.

---

### temporal_dataset.py

The temporal-branch equivalent of `image_loader.py`. Loads the cached `.npy` sequences using the same `train.csv`/`val.csv`/`test.csv` splits, so both branches train/validate/test on the exact same videos.

---

# Semantic Feature Extraction

Current backbone:

**EfficientNet-B0**

Why EfficientNet?

- Lightweight
- High accuracy
- Transfer learning support
- Suitable for deployment
- Faster than training a custom CNN from scratch

Architecture

```
Input Face Image
        │
        ▼
EfficientNet-B0
        │
        ▼
Feature Vector
        │
        ▼
Fully Connected Layers
        │
        ▼
Binary Classification
```

Output

```
Real
Fake
```

---

# Temporal Feature Extraction

Current architecture: **CNN (ResNet-18, frozen) + LSTM**

Chosen over Optical Flow+CNN, ConvLSTM, 3D CNN, and Video Transformer as the best accuracy/compute/deployment trade-off for a Colab-compatible, capstone-scale project — see `src/features/temporal_extractor.py` for the full architecture rationale in code comments.

Architecture

```
16-Frame Face Sequence
        │
        ▼
ResNet-18 (per frame, frozen)
        │
        ▼
Per-Frame Feature Sequence [16, 512]
        │
        ▼
LSTM
        │
        ▼
Temporal Embedding [256]
        │
        ▼
Fully Connected Layer
        │
        ▼
Binary Classification
```

Both the semantic and temporal branches expose a 256-dimensional embedding (`embedding, logits = model(x)`), by design — this is what Feature Fusion (Phase 3) will concatenate into a 512-dimensional fused vector.

---

# Training Pipeline

Current pipeline

```
Face Images
        │
        ▼
Image Loader
        │
        ▼
EfficientNet-B0
        │
        ▼
CrossEntropy Loss
        │
        ▼
Adam Optimizer
```

Training includes

- Mixed Precision Training (AMP)
- Train → Validation each epoch, with the best checkpoint selected on **validation accuracy** (the test set is never used for model selection)
- Automatic checkpoint saving

The temporal branch's training additionally includes:

- Early stopping (patience-based)
- Learning-rate scheduling (`ReduceLROnPlateau`)
- Progress bars and file-based logging (`saved_models/train_temporal.log`)

---

# Evaluation

The test set is evaluated exactly once, after model selection, never during training.

Metrics reported (both branches)

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report
- ROC-AUC (temporal branch)

Results are printed to console and saved as JSON:

```
saved_models/test_evaluation_report.json           # semantic branch
saved_models/test_temporal_evaluation_report.json  # temporal branch
```

Model checkpoints are stored as

```
saved_models/
├── semantic_checkpoint.pth
└── temporal_checkpoint.pth
```

---

# Technologies Used

### Programming Language

- Python

### Deep Learning

- PyTorch
- Torchvision

### Computer Vision

- OpenCV

### Data Handling

- Pandas
- NumPy

### Machine Learning

- Scikit-learn

### Dataset

- Celeb-DF v2
- KaggleHub

---

# Current Progress

## Completed

- Project structure
- Dataset setup
- CSV-based train/validation/test split pipeline
- Frame sampling
- Face detection (MTCNN)
- Face preprocessing (semantic + temporal, both resumable)
- EfficientNet-B0 semantic classifier, exposing embeddings
- CNN (ResNet-18) + LSTM temporal classifier, exposing embeddings
- Validation-based checkpointing (both branches)
- Full evaluation metrics incl. ROC-AUC
- Model checkpointing

---

## In Progress

- Feature Fusion design

---

## Upcoming

- Feature Fusion implementation
- Final Hybrid Model
- Video Inference Pipeline
- REST API
- Web Interface
- Confidence Score Visualization

---

# Future Pipeline

```
Uploaded Video
        │
        ▼
Frame Sampling
        │
        ▼
Face Detection
        │
        ▼
Semantic Features
        │
        ▼
Temporal Features
        │
        ▼
Feature Fusion
        │
        ▼
Classifier
        │
        ▼
Prediction
```

---

# Planned Web Application

The final system will allow users to:

- Upload a video
- Process uploaded frames
- Detect manipulated facial regions
- Predict whether the video is Real or Fake
- Display confidence score
- View inference results in real time

---

# Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/hybrid-deepfake-detector.git
```

Navigate into the project

```bash
cd hybrid-deepfake-detector
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download dataset

```python
import kagglehub

kagglehub.dataset_download("reubensuju/celeb-df-v2")
```

Run preprocessing

```bash
python -m src.preprocessing.setup_dataset
python -m src.preprocessing.create_splits
python -m src.preprocessing.precompute_faces
python -m src.preprocessing.precompute_temporal
```

Train and evaluate the semantic branch

```bash
python -m src.modeling.train_semantic
python -m src.modeling.test_semantic
```

Train and evaluate the temporal branch

```bash
python -m src.modeling.train_temporal
python -m src.modeling.test_temporal
```

---

# Roadmap

- [x] Project setup
- [x] Dataset preparation
- [x] Frame sampling
- [x] Face detection
- [x] Semantic feature extraction
- [x] Semantic model training
- [x] Semantic model evaluation
- [x] Temporal feature extraction
- [ ] Feature fusion
- [ ] Hybrid classifier
- [ ] Video inference
- [ ] REST API
- [ ] Web interface
- [ ] Deployment

---

# Authors

**Team**

- Hardik Abrol
- Gaurang Mangla
- Kaushik Arora
- Sarthak Gaba
- Yash Sidana

B.Tech Capstone Project

Hybrid Deepfake Detection using Semantic and Temporal Feature Fusion
