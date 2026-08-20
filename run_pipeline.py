"""
Hybrid Deepfake Detector — Unified Local Pipeline Runner
========================================================
Runs the end-to-end multi-modal deepfake detection pipeline on local GPU:
1. Dataset Download & Verification (Celeb-DF v2 + DFD)
2. Metadata Extraction & Subsetting
3. Identity-Aware Leakage-Safe Splits
4. Feature Precomputation (Semantic Faces, Optical Flow Sequences, Forensic Features)
5. Semantic Branch Training (EfficientNet-B0)
6. Temporal Branch Training (ResNet-18 + LSTM)
7. Embedding Extraction (575-D Multi-Modal Representations)
8. Fusion Classifier Training (Distribution Matching + RBF SVM)
9. Final Test Set Evaluation
10. Ablation Study & Cross-Dataset Generalization

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --stage download
    python run_pipeline.py --stage preprocess
    python run_pipeline.py --stage train
    python run_pipeline.py --stage evaluate
    python run_pipeline.py --stage ablation
    python run_pipeline.py --stage cross-dataset
"""

import argparse
import os
import sys
import time
import subprocess
import json
import torch

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def check_gpu():
    print_header("NVIDIA GPU Diagnostics")
    if not torch.cuda.is_available():
        print("WARNING: CUDA is NOT available! Running on CPU will be significantly slower.")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cuda_ver = torch.version.cuda
    print(f"  GPU Device:      {device_name}")
    print(f"  Total VRAM:      {vram_gb:.2f} GB")
    print(f"  PyTorch CUDA:    {cuda_ver}")
    print(f"  Compute Cap:     {torch.cuda.get_device_capability(0)}")
    return True

def ensure_dirs():
    dirs = [
        "data/raw",
        "data/processed",
        "data/metadata",
        "data/splits",
        "models",
        "saved_models"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Workspace directories verified.")

def run_module(module_name, args=None):
    cmd = [sys.executable, "-m", module_name]
    if args:
        cmd.extend(args)
    print(f"Running: {' '.join(cmd)}")
    start = time.time()
    res = subprocess.run(cmd, check=False)
    elapsed = time.time() - start
    print(f"Finished {module_name} in {elapsed:.2f}s (Exit code: {res.returncode})\n")
    if res.returncode != 0:
        raise RuntimeError(f"Step {module_name} failed with exit code {res.returncode}")

def check_kaggle_creds():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json):
        print(f"Kaggle credentials found at {kaggle_json}")
        return True
    
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        print(f"Kaggle credentials found in environment variables (User: {username})")
        return True
    
    print("\n" + "!" * 70)
    print("  Kaggle credentials NOT found!")
    print("  Please place your 'kaggle.json' in:")
    print(f"    {kaggle_json}")
    print("  Or set environment variables KAGGLE_USERNAME and KAGGLE_KEY.")
    print("!" * 70 + "\n")
    return False

def stage_download():
    print_header("Stage 1: Dataset Download (Celeb-DF v2 + DFD)")
    if not check_kaggle_creds():
        raise FileNotFoundError("Kaggle credentials missing. See prompt above.")
    
    from src.data.adapters.celebdf import CelebDFAdapter
    from src.data.adapters.dfd import DFDAdapter
    
    print("[1/2] Downloading / Verifying Celeb-DF v2...")
    CelebDFAdapter().download()
    
    print("[2/2] Downloading / Verifying DFD...")
    DFDAdapter().download()
    print("Datasets downloaded and organized under data/raw.")

def stage_metadata():
    print_header("Stage 2: Metadata Generation & Subsetting")
    run_module("src.data.metadata")
    run_module("src.preprocessing.subset_metadata", ["--target", "celebdf:1000:1000", "--target", "dfd:1000:1000"])
    run_module("src.preprocessing.create_splits")

def stage_precompute():
    print_header("Stage 3: Feature Precomputation (Cached & Resumable)")
    print("[1/3] Precomputing Semantic Face Crops...")
    run_module("src.preprocessing.precompute_faces")
    
    print("[2/3] Precomputing Temporal Optical Flow Sequences...")
    run_module("src.preprocessing.precompute_temporal")
    
    print("[3/3] Precomputing Forensic Features (rPPG + Frequency FFT)...")
    run_module("src.preprocessing.precompute_forensic")

def stage_train_branches():
    print_header("Stage 4: Training Semantic & Temporal Deep Learning Branches")
    print("[1/2] Training Semantic Branch (EfficientNet-B0)...")
    run_module("src.modeling.train_semantic")
    
    print("[2/2] Training Temporal Branch (ResNet-18 + LSTM)...")
    run_module("src.modeling.train_temporal")

def stage_fusion():
    print_header("Stage 5: Embedding Extraction & Fusion SVM Training")
    print("[1/3] Extracting 575-D Multi-Modal Embeddings...")
    run_module("src.modeling.extract_embeddings")
    
    print("[2/3] Training Distribution Matcher + RBF SVM Fusion Classifier...")
    run_module("src.modeling.train_fusion")
    
    print("[3/3] Running Final Test Set Evaluation...")
    run_module("src.modeling.test_fusion")

def stage_ablation():
    print_header("Stage 6: Ablation Study (Branch Importance)")
    run_module("src.modeling.run_ablation")

def stage_cross_dataset():
    print_header("Stage 7: Cross-Dataset Generalization")
    print("Direction 1: Train on DFD -> Test on Celeb-DF v2")
    run_module("src.preprocessing.create_cross_dataset_splits", [
        "--train-dataset", "dfd", "--test-dataset", "celebdf",
        "--out-dir", "data/splits_cross_dfd_to_celebdf"
    ])
    run_module("src.modeling.extract_embeddings", [
        "--splits-root", "data/splits_cross_dfd_to_celebdf",
        "--output-root", "data/processed/fusion_cross_dfd_to_celebdf"
    ])
    run_module("src.modeling.train_fusion", [
        "--embeddings-root", "data/processed/fusion_cross_dfd_to_celebdf",
        "--model-path", "models/fusion_classifier/fusion_model_cross_dfd_to_celebdf.pkl",
        "--report-path", "saved_models/train_fusion_report_cross_dfd_to_celebdf.json"
    ])
    run_module("src.modeling.test_fusion", [
        "--embeddings-root", "data/processed/fusion_cross_dfd_to_celebdf",
        "--model-path", "models/fusion_classifier/fusion_model_cross_dfd_to_celebdf.pkl",
        "--report-path", "saved_models/test_fusion_evaluation_report_cross_dfd_to_celebdf.json"
    ])

def main():
    parser = argparse.ArgumentParser(description="Hybrid Deepfake Detector Local Pipeline Runner")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline end-to-end")
    parser.add_argument("--stage", choices=[
        "check", "download", "metadata", "preprocess",
        "train", "fusion", "evaluate", "ablation", "cross-dataset"
    ], help="Run a specific stage")
    args = parser.parse_args()

    check_gpu()
    ensure_dirs()

    if args.stage == "check":
        print("Pre-flight check passed.")
        return

    if args.all or args.stage == "download":
        stage_download()
    if args.all or args.stage == "metadata":
        stage_metadata()
    if args.all or args.stage == "preprocess":
        stage_precompute()
    if args.all or args.stage == "train":
        stage_train_branches()
    if args.all or args.stage in ("fusion", "evaluate"):
        stage_fusion()
    if args.all or args.stage == "ablation":
        stage_ablation()
    if args.all or args.stage == "cross-dataset":
        stage_cross_dataset()

    print_header("Pipeline Execution Completed Successfully!")

if __name__ == "__main__":
    main()
