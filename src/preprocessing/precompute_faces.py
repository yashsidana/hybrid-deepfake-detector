"""
One-time preprocessing: samples a single key frame per video, runs face
detection, and caches the result to disk. Training reads these cached
images instead of touching raw video ever again.

Reads from data/metadata/metadata.csv (the multi-dataset source of truth)
instead of a hardcoded dataset folder -- this is what makes it work
automatically across every enabled dataset, with zero dataset-specific
logic here. Output preserves each dataset's own relative path structure
under a per-dataset namespace, so different datasets' videos never collide
even if they happen to share a filename.

    data/processed/semantic/<dataset>/<video_path with .mp4 -> .jpg>
"""

import os

import cv2
import pandas as pd
from tqdm import tqdm

from src.preprocessing.face_detection import detect_face
from src.preprocessing.frame_sampler import sample_frames

METADATA_PATH = "data/metadata/metadata.csv"
RAW_ROOT = "data/raw"
SAVE_ROOT = "data/processed/semantic"


def process_all(metadata_path=METADATA_PATH, raw_root=RAW_ROOT, save_root=SAVE_ROOT):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found. Run `python -m src.data.metadata` first."
        )

    df = pd.read_csv(metadata_path)
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing semantic faces"):
        dataset = row["dataset"]
        video_rel_path = row["video_path"]

        video_abs_path = os.path.join(raw_root, dataset, video_rel_path)
        save_path = os.path.join(save_root, dataset, video_rel_path.replace(".mp4", ".jpg"))

        if os.path.exists(save_path):
            continue  # already cached from a previous run -- resumable

        if not os.path.exists(video_abs_path):
            skipped += 1
            continue  # metadata says it should exist but it doesn't -- see validate.py

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            frames = sample_frames(video_abs_path, num_frames=1)
            frame = frames[0]
            face = detect_face(frame)
            cv2.imwrite(save_path, face)
        except Exception as e:
            skipped += 1
            print(f"  [skip] {dataset}/{video_rel_path}: {e}")

    if skipped:
        print(f"Skipped {skipped} video(s) -- see above / run validate.py for details.")


if __name__ == "__main__":
    process_all()
