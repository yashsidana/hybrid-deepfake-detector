# Hybrid Deepfake Detector

A modular AI-based Hybrid Deepfake Detection System that combines **Semantic Feature Extraction**, **Temporal Feature Extraction**, and **Feature Fusion** to accurately classify videos as **Real** or **Fake**.

> **Current Status:** Phase 2 (Temporal Feature Extraction) Completed + Multi-Dataset Data Pipeline Completed — Feature Fusion next

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

- Multi-dataset support (Celeb-DF v2, DFDC, with FaceForensics++/FakeAVCeleb/GenVidBench/custom also wired in — see Dataset section)
- Identity-aware, leakage-safe train/validation/test splitting (union-find grouping + label-stratified)
- Class-imbalance-aware training (weighted loss + macro-F1 model selection)
- Automatic train/validation/test split generation
- Video frame sampling
- Face detection preprocessing (MTCNN, GPU-aware)
- Face image + sequence caching for faster training (both resumable)
- EfficientNet-B0 semantic classifier + CNN(ResNet-18)-LSTM temporal classifier, both exposing embeddings
- Mixed Precision (AMP) Training
- Checkpoint saving
- Full evaluation metrics (accuracy, precision, recall, F1, macro-F1, balanced accuracy, ROC-AUC, confusion matrix)

---

# Repository Structure

```
hybrid-deepfake-detector/
│
├── config/
│
├── data/
│   ├── raw/
│   │   ├── celebdf/
│   │   ├── dfdc/
│   │   ├── faceforensics/       (manual setup — see Dataset section)
│   │   ├── fakeavceleb/          (manual setup — see Dataset section)
│   │   ├── genvidbench/
│   │   └── custom/
│   ├── processed/
│   │   ├── semantic/<dataset>/  (precomputed face images)
│   │   └── temporal/<dataset>/  (precomputed 16-frame sequences)
│   ├── metadata/
│   │   ├── metadata.csv         (single source of truth across all enabled datasets)
│   │   └── validation_report.json
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   ├── data/                    (multi-dataset collection — see Dataset section)
│   │   ├── base_dataset.py
│   │   ├── registry.py
│   │   ├── metadata.py
│   │   ├── validate.py
│   │   └── adapters/
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

# Dataset Collection Architecture

Datasets are integrated through a common adapter interface
(`src/data/base_dataset.py`) so adding a new dataset never requires
touching the rest of the codebase — one adapter file, one line in the
registry. Every downstream step (splitting, semantic/temporal
preprocessing) talks only to `data/metadata/metadata.csv`, never to a raw
dataset folder directly.

| Dataset | Status | Access |
|---|---|---|
| Celeb-DF v2 | ✅ Enabled by default | Fully automated (KaggleHub) |
| DFDC | ✅ Enabled | Automated after a one-time manual step: accept the [competition rules](https://www.kaggle.com/c/deepfake-detection-challenge/rules) on Kaggle. Uses the official sample set (`train_sample_videos.zip`) by default — full 470GB set available via `part_index`. |
| FaceForensics++ | Wired in, disabled by default | **Manual only** — requires filling out [the official access form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform) and waiting for approval. No API exists; this is the dataset owners' policy, not a limitation of this project. |
| FakeAVCeleb | Wired in, disabled by default | **Manual only** — same pattern, [official request form](https://bit.ly/38prlVO) + Data Use Agreement review. |
| GenVidBench | Wired in, disabled by default | Publicly downloadable (HuggingFace), but requires an explicit `hf_repo_id` in `config.yaml` — no single canonical repo ID could be confirmed at time of writing, so this is left to the user to verify at [genvidbench.github.io](https://genvidbench.github.io/) rather than guessed. |
| Custom | Wired in, disabled by default | Drop videos into `data/raw/custom/{real,fake}/` — no code changes needed. |

Enable/disable datasets in `config.yaml`:

```yaml
datasets:
  celebdf: true
  dfdc: true
  faceforensics: false
  fakeavceleb: false
  genvidbench: false
  custom: false
```

### Identity-aware splitting

`create_splits.py` uses a union-find over each video's identity token(s)
(comma-separated for videos referencing multiple identities, e.g. a
source+target face swap) so that any two videos sharing an identity are
transitively grouped and never land on opposite sides of the
train/val/test boundary — this was a known gap in earlier versions of this
project (video-level-only splitting) and is now enforced, with an internal
sanity check that raises an error rather than silently writing leaked
splits. Videos with no identity information (e.g. plain custom uploads)
fall back to being their own singleton group, equivalent to a video-level
split for exactly those rows.

---

# Dataset

Currently enabled datasets: **Celeb-DF v2** and **DFDC** (sample set).

Dataset is downloaded automatically using KaggleHub.

```python
import kagglehub

path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
```

Dataset Structure

```
data/
└── raw/
    ├── celebdf/
    │   ├── real/
    │   └── fake/
    └── dfdc/
        └── train_sample_videos/
            ├── metadata.json
            └── *.mp4
```

Each additional enabled dataset gets its own subfolder under `data/raw/`,
preserving its own native directory hierarchy rather than being flattened
into a shared structure — see the Dataset Collection Architecture section
above for the full list and access requirements per dataset.

---

# Preprocessing Pipeline

The preprocessing module converts raw videos into training-ready face images, across every enabled dataset.

Pipeline:

```
Raw Videos (any enabled dataset)
        │
        ▼
src/data/metadata.py           (scans every enabled adapter, writes metadata.csv)
        │
        ▼
src/data/validate.py           (pre-flight validation report)
        │
        ▼
create_splits.py               (identity-aware, multi-dataset, label-stratified)
        │
        ▼
frame_sampler.py
        │
        ▼
face_detection.py
        │
        ▼
precompute_faces.py / precompute_temporal.py
        │
        ▼
data/processed/semantic|temporal/<dataset>/
        │
        ▼
image_loader.py / temporal_dataset.py
```

### Preprocessing Modules

### src/data/ (dataset collection — see Dataset Collection Architecture above)

- `base_dataset.py` — `BaseDatasetAdapter` interface + `VideoRecord` schema every adapter produces
- `registry.py` — reads `config.yaml`'s `datasets:` section, instantiates only enabled adapters
- `metadata.py` — calls every enabled adapter's `list_videos()`, writes `data/metadata/metadata.csv`
- `validate.py` — checks the metadata for missing files, corrupted videos, duplicate rows, incorrect labels, invalid frame counts; writes `data/metadata/validation_report.json`
- `adapters/` — one file per dataset (celebdf, dfdc, faceforensics, fakeavceleb, genvidbench, custom)

---

### setup_dataset.py

- Downloads Celeb-DF using KaggleHub
- Organizes videos into Real/Fake folders
- (Other datasets are downloaded via their own adapter — see `src/data/adapters/`)

---

### create_splits.py

Reads `data/metadata/metadata.csv` (across every enabled dataset) and creates

- Train set
- Validation set
- Test set

using an identity-aware, label-stratified 80-10-10 split (`StratifiedGroupKFold` + union-find over identity tokens — see Dataset Collection Architecture above). Output CSVs include `video_path`, `dataset`, `identity`, `label`.

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

One-time preprocessing step. Reads `data/metadata/metadata.csv` (not a hardcoded dataset path), so it works automatically across every enabled dataset with zero dataset-specific logic.

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

Output is namespaced per dataset to avoid collisions: `data/processed/semantic/<dataset>/<original relative path>.jpg`. Training is performed on these precomputed face images instead of videos, significantly reducing training time. Resumable — already-cached videos are skipped on a rerun.

---

### image_loader.py

Uses a custom CSV-driven `Dataset` that reads `data/splits/train.csv`, `val.csv`, and `test.csv` — no random re-splitting, so the same train/val/test boundaries are used everywhere in the pipeline. Resolves each row's `(dataset, video_path)` pair to its namespaced precomputed face image.

Returns

- Train DataLoader
- Validation DataLoader
- Test DataLoader

---

### precompute_temporal.py

One-time preprocessing step for the temporal branch — same metadata-driven, multi-dataset design as `precompute_faces.py`.

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

Output: `data/processed/temporal/<dataset>/<original relative path>.npy`. Resumable — already-cached videos are skipped on a rerun, so an interrupted preprocessing run can pick up where it left off.

---

### temporal_dataset.py

The temporal-branch equivalent of `image_loader.py`. Loads the cached `.npy` sequences using the same `train.csv`/`val.csv`/`test.csv` splits and `(dataset, video_path)` resolution, so both branches train/validate/test on the exact same videos regardless of how many datasets are enabled.

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
Class-Weighted CrossEntropy Loss
        │
        ▼
Adam Optimizer
```

Training includes

- Mixed Precision Training (AMP)
- **Class-weighted loss** — Celeb-DF (and DFDC) are heavily imbalanced toward "fake" (~86%/14%); inverse-frequency class weights are computed from the training set each run and applied to `CrossEntropyLoss`, so misclassifying the minority "real" class costs proportionally more. Without this, the model can (and, in an earlier unweighted run, did) collapse to predicting "fake" almost always — ~86% accuracy while catching only 3% of real videos.
- Train → Validation each epoch, with the best checkpoint selected on **validation macro-F1** (not raw accuracy — accuracy alone rewards exactly the majority-class collapse above; macro-F1 is the average of each class's own F1, so it can't be gamed by ignoring the minority class). Balanced accuracy is also computed and logged as an easy sanity check. The test set is never used for model selection.
- Automatic checkpoint saving (guaranteed to save at least once per run, even in a degenerate all-zero-F1 edge case)

The temporal branch's training additionally includes:

- Early stopping (patience-based, tracking macro-F1)
- Learning-rate scheduling (`ReduceLROnPlateau`, tracking macro-F1)
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
- Classification Report (per-class precision/recall/F1 — check this, not just top-line accuracy, given the dataset's class imbalance)
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
- DFDC
- KaggleHub
- HuggingFace Hub (for GenVidBench)

---

# Current Progress

## Completed

- Project structure
- Multi-dataset collection architecture (adapter interface, registry, metadata generation, validation) — Celeb-DF v2 and DFDC enabled
- Identity-aware, leakage-safe, multi-dataset train/validation/test split pipeline
- Frame sampling
- Face detection (MTCNN, GPU-aware)
- Face preprocessing (semantic + temporal, both resumable, dataset-namespaced)
- EfficientNet-B0 semantic classifier, exposing embeddings
- CNN (ResNet-18) + LSTM temporal classifier, exposing embeddings
- Class-imbalance-aware training (weighted loss, macro-F1 model selection) for both branches
- Full evaluation metrics incl. macro-F1, balanced accuracy, ROC-AUC
- Model checkpointing

---

## In Progress

- Retraining both branches under the new multi-dataset, identity-aware splits (previous checkpoints/metrics were produced under the older, non-identity-aware Celeb-DF-only split and should be treated as superseded once this completes)
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

Configure which datasets are enabled in `config.yaml` (`celebdf` and `dfdc` are enabled by default):

```yaml
datasets:
  celebdf: true
  dfdc: true
  faceforensics: false
  fakeavceleb: false
  genvidbench: false
  custom: false
```

Download enabled datasets — Celeb-DF is automated:

```python
import kagglehub

kagglehub.dataset_download("reubensuju/celeb-df-v2")
```

DFDC requires accepting the [competition rules](https://www.kaggle.com/c/deepfake-detection-challenge/rules) once in a browser first, then:

```python
from src.data.adapters.dfdc import DFDCAdapter
DFDCAdapter().download(part_index="sample")
```

FaceForensics++/FakeAVCeleb require manual approval — see the Dataset Collection Architecture section above for the official request forms and expected folder layout once you have the data. GenVidBench and custom datasets are documented there too.

Generate metadata, validate, and create splits (across every enabled dataset):

```bash
python -m src.data.metadata
python -m src.data.validate
python -m src.preprocessing.create_splits
```

Run preprocessing

```bash
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
- [x] Multi-dataset collection architecture (Celeb-DF v2 + DFDC enabled)
- [x] Identity-aware, leakage-safe splitting
- [x] Frame sampling
- [x] Face detection
- [x] Semantic feature extraction
- [x] Semantic model training (class-imbalance-aware)
- [x] Semantic model evaluation
- [x] Temporal feature extraction
- [ ] Retrain both branches under new multi-dataset splits
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
