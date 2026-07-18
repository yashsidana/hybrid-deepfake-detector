# Hybrid Deepfake Detector

A modular AI-based Hybrid Deepfake Detection System that combines **Semantic Feature Extraction**, **Temporal Feature Extraction**, and **Feature Fusion** to accurately classify videos as **Real** or **Fake**.

> **Current Status:** Phase 1 (Semantic Feature Extraction) Completed

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
Temporal Feature Extraction (Upcoming)
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

Uses

```
torchvision.datasets.ImageFolder
```

to load face images.

Returns

- Train DataLoader
- Test DataLoader

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
- Automatic checkpoint saving
- Best model preservation

---

# Evaluation

Current evaluation metrics

- Training Accuracy
- Test Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Model checkpoints are stored as

```
saved_models/
└── semantic_checkpoint.pth
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
- Data preprocessing
- Frame sampling
- Face detection
- Face preprocessing
- EfficientNet-B0 semantic classifier
- Training pipeline
- Evaluation pipeline
- Model checkpointing

---

## In Progress

- Temporal Feature Extraction
- Optical Flow generation
- Motion feature learning

---

## Upcoming

- Feature Fusion
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
```

Train semantic model

```bash
python -m src.modeling.train_semantic
```

Evaluate model

```bash
python -m src.modeling.test_semantic
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
- [ ] Temporal feature extraction
- [ ] Feature fusion
- [ ] Hybrid classifier
- [ ] Video inference
- [ ] REST API
- [ ] Web interface
- [ ] Deployment

---

# Authors

**Yash Sidana**

B.Tech Capstone Project

Hybrid Deepfake Detection using Semantic and Temporal Feature Fusion
