"""
One-time preprocessing: computes the handcrafted forensic feature vector
(SRM + statistical texture + landmark motion + rPPG -- see
src/features/forensic_extractor.py) for every video and caches it to disk.
Same "precompute once, train many times" pattern as precompute_faces.py
and precompute_temporal.py.

Requires precompute_faces.py and precompute_temporal.py to have already
run -- SRM/texture/landmark-motion reuse their cached outputs instead of
re-detecting faces from raw video. rPPG is the one exception: it reads the
raw video directly for its own denser frame sampling (see
forensic_extractor.py's rppg_features() docstring for why).

    data/processed/forensic/<dataset>/<video_path with .mp4 -> .npy>
"""

import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.forensic_extractor import extract_forensic_vector

from concurrent.futures import ThreadPoolExecutor, as_completed

METADATA_PATH = "data/metadata/metadata.csv"
RAW_ROOT = "data/raw"
FACES_ROOT = "data/processed/semantic"
SEQUENCES_ROOT = "data/processed/temporal"
SAVE_ROOT = "data/processed/forensic"


def _process_single(row, raw_root, faces_root, sequences_root, save_root):
    dataset = row["dataset"]
    video_rel_path = row["video_path"]

    face_path = os.path.join(faces_root, dataset, video_rel_path.replace(".mp4", ".jpg"))
    seq_path = os.path.join(sequences_root, dataset, video_rel_path.replace(".mp4", ".npy"))
    video_abs_path = os.path.join(raw_root, dataset, video_rel_path)
    save_path = os.path.join(save_root, dataset, video_rel_path.replace(".mp4", ".npy"))

    if os.path.exists(save_path):
        return "cached"

    if not (os.path.exists(face_path) and os.path.exists(seq_path) and os.path.exists(video_abs_path)):
        return "not_ready"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        face_image = cv2.imread(face_path)
        face_sequence = np.load(seq_path)
        if face_image is None:
            return "skipped"

        vector = extract_forensic_vector(video_abs_path, face_image, face_sequence)
        np.save(save_path, vector)
        return "done"
    except Exception as e:
        return "skipped"


def process_all(metadata_path=METADATA_PATH, raw_root=RAW_ROOT, faces_root=FACES_ROOT,
                 sequences_root=SEQUENCES_ROOT, save_root=SAVE_ROOT, max_workers=8):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found. Run `python -m src.data.metadata` first."
        )

    df = pd.read_csv(metadata_path)
    rows = [row for _, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_single, row, raw_root, faces_root, sequences_root, save_root)
            for row in rows
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Precomputing forensic features (multi-threaded)"):
            f.result()

    print(f"\nForensic feature vectors cached to: {save_root}")


if __name__ == "__main__":
    process_all()
