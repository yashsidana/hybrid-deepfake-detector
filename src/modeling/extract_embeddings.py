"""
Phase 3 (Feature Fusion) step 1: run the trained semantic and temporal
branches over every video in train/val/test and cache their embeddings.

This is the bridge between the two already-trained branches
(train_semantic.py / train_temporal.py) and the fusion classifier
(train_fusion.py) — everything downstream of this script works on cached
[N, 256] embedding arrays instead of touching raw videos or images again,
the same "precompute once, train many times" pattern precompute_faces.py
and precompute_temporal.py already established for the branches themselves.

Only videos with BOTH a precomputed face image (semantic) AND a
precomputed face sequence (temporal) are included — a row missing either
one is dropped and counted, since fusion structurally needs both branches'
embeddings for the same video.

Output: data/processed/fusion/<split>.npz, each containing:
    semantic_embeddings   [N, 256] float32
    temporal_embeddings   [N, 256] float32
    labels                [N]      int64
    video_paths           [N]      object (str) — "<dataset>/<video_path>", for traceability
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.features.semantic_extractor import SemanticClassifier
from src.features.temporal_extractor import TemporalClassifier

SPLITS_ROOT = "data/splits"
FACES_ROOT = "data/processed/semantic"
SEQUENCES_ROOT = "data/processed/temporal"
OUTPUT_ROOT = "data/processed/fusion"

SEMANTIC_CHECKPOINT = "saved_models/semantic_checkpoint.pth"
TEMPORAL_CHECKPOINT = "saved_models/temporal_checkpoint.pth"

_semantic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Same ImageNet normalization temporal_dataset.py uses, since the temporal
# branch's per-frame backbone (ResNet-18) is ImageNet-pretrained.
_temporal_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def _load_semantic_model(device):
    if not os.path.exists(SEMANTIC_CHECKPOINT):
        raise FileNotFoundError(
            f"No semantic checkpoint at {SEMANTIC_CHECKPOINT}. Run "
            f"`python -m src.modeling.train_semantic` first."
        )
    model = SemanticClassifier().to(device)
    checkpoint = torch.load(SEMANTIC_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_temporal_model(device):
    if not os.path.exists(TEMPORAL_CHECKPOINT):
        raise FileNotFoundError(
            f"No temporal checkpoint at {TEMPORAL_CHECKPOINT}. Run "
            f"`python -m src.modeling.train_temporal` first."
        )
    model = TemporalClassifier().to(device)
    checkpoint = torch.load(TEMPORAL_CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _load_semantic_input(dataset, video_path):
    stem = video_path.replace(".mp4", ".jpg")
    face_path = os.path.join(FACES_ROOT, dataset, stem)
    if not os.path.exists(face_path):
        return None
    image = Image.open(face_path).convert("RGB")
    return _semantic_transform(image)


def _load_temporal_input(dataset, video_path):
    stem = video_path.replace(".mp4", ".npy")
    seq_path = os.path.join(SEQUENCES_ROOT, dataset, stem)
    if not os.path.exists(seq_path):
        return None
    sequence = np.load(seq_path)
    sequence = sequence[..., ::-1].copy()  # BGR -> RGB, matches temporal_dataset.py's contract
    sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2).float() / 255.0
    sequence = torch.stack([_temporal_normalize(frame) for frame in sequence])
    return sequence


@torch.no_grad()
def extract_split(split_name, semantic_model, temporal_model, device,
                   splits_root=SPLITS_ROOT, batch_size=16):
    csv_path = os.path.join(splits_root, f"{split_name}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python -m src.preprocessing.create_splits` first."
        )

    df = pd.read_csv(csv_path)

    semantic_chunks, temporal_chunks = [], []
    labels, video_paths = [], []
    skipped = 0

    semantic_batch, temporal_batch, label_batch, path_batch = [], [], [], []

    def _flush():
        nonlocal semantic_batch, temporal_batch, label_batch, path_batch
        if not semantic_batch:
            return
        sem_tensor = torch.stack(semantic_batch).to(device)
        temp_tensor = torch.stack(temporal_batch).to(device)

        sem_emb, _ = semantic_model(sem_tensor)
        temp_emb, _ = temporal_model(temp_tensor)

        semantic_chunks.append(sem_emb.cpu().numpy())
        temporal_chunks.append(temp_emb.cpu().numpy())
        labels.extend(label_batch)
        video_paths.extend(path_batch)

        semantic_batch, temporal_batch, label_batch, path_batch = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Embedding [{split_name}]"):
        dataset, video_path, label = row["dataset"], row["video_path"], int(row["label"])

        semantic_input = _load_semantic_input(dataset, video_path)
        temporal_input = _load_temporal_input(dataset, video_path)

        if semantic_input is None or temporal_input is None:
            skipped += 1
            continue

        semantic_batch.append(semantic_input)
        temporal_batch.append(temporal_input)
        label_batch.append(label)
        path_batch.append(f"{dataset}/{video_path}")

        if len(semantic_batch) >= batch_size:
            _flush()

    _flush()

    if skipped:
        print(
            f"[{split_name}] Skipped {skipped} video(s) missing a "
            f"precomputed semantic face and/or temporal sequence."
        )

    if not semantic_chunks:
        raise RuntimeError(
            f"No usable samples for split '{split_name}' — every row was "
            f"missing precomputed semantic and/or temporal data. Run "
            f"precompute_faces.py and precompute_temporal.py first."
        )

    return {
        "semantic_embeddings": np.concatenate(semantic_chunks, axis=0),
        "temporal_embeddings": np.concatenate(temporal_chunks, axis=0),
        "labels": np.array(labels, dtype=np.int64),
        "video_paths": np.array(video_paths, dtype=object),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    semantic_model = _load_semantic_model(device)
    temporal_model = _load_temporal_model(device)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for split_name in ("train", "val", "test"):
        result = extract_split(split_name, semantic_model, temporal_model, device)
        out_path = os.path.join(OUTPUT_ROOT, f"{split_name}.npz")
        np.savez(
            out_path,
            semantic_embeddings=result["semantic_embeddings"],
            temporal_embeddings=result["temporal_embeddings"],
            labels=result["labels"],
            video_paths=result["video_paths"],
        )
        print(
            f"[{split_name}] {len(result['labels'])} embeddings saved to {out_path} "
            f"(semantic {result['semantic_embeddings'].shape}, "
            f"temporal {result['temporal_embeddings'].shape})"
        )

    print("\nRun `python -m src.modeling.train_fusion` next.")


if __name__ == "__main__":
    main()
