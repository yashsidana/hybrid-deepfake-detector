"""
One-time preprocessing: samples 16 frames per video, face-detects each one,
and caches the resulting sequence to disk as a single .npy file per video.

Same multi-dataset design as precompute_faces.py: reads from
data/metadata/metadata.csv, writes under a per-dataset namespace so no
dataset-specific logic lives here.

    data/processed/temporal/<dataset>/<video_path with .mp4 -> .npy>
"""

import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.face_detection import detect_face
from src.preprocessing.frame_sampler import sample_frames

METADATA_PATH = "data/metadata/metadata.csv"
RAW_ROOT = "data/raw"
SAVE_ROOT = "data/processed/temporal"
NUM_FRAMES = 16


def process_all(metadata_path=METADATA_PATH, raw_root=RAW_ROOT, save_root=SAVE_ROOT, num_frames=NUM_FRAMES):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found. Run `python -m src.data.metadata` first."
        )

    df = pd.read_csv(metadata_path)
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing temporal sequences"):
        dataset = row["dataset"]
        video_rel_path = row["video_path"]

        video_abs_path = os.path.join(raw_root, dataset, video_rel_path)
        save_path = os.path.join(save_root, dataset, video_rel_path.replace(".mp4", ".npy"))

        if os.path.exists(save_path):
            continue  # already cached from a previous run -- resumable

        if not os.path.exists(video_abs_path):
            skipped += 1
            continue

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            frames = sample_frames(video_abs_path, num_frames=num_frames, size=(224, 224))
            face_sequence = np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
            for i in range(frames.shape[0]):
                face_sequence[i] = detect_face(frames[i])
            np.save(save_path, face_sequence)
        except Exception as e:
            skipped += 1
            print(f"  [skip] {dataset}/{video_rel_path}: {e}")

    if skipped:
        print(f"Skipped {skipped} video(s) -- see above / run validate.py for details.")
    print(f"\nTemporal face sequences cached to: {save_root}")


if __name__ == "__main__":
    process_all()
