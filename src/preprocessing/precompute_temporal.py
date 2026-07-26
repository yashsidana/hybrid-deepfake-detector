import os

import numpy as np
from tqdm import tqdm

from src.preprocessing.face_detection import detect_face
from src.preprocessing.frame_sampler import sample_frames

DATASET_ROOT = "data/raw/celebdf"
SAVE_ROOT = "data/processed/temporal_faces"
NUM_FRAMES = 16


def process_class(class_name, num_frames=NUM_FRAMES):
    input_dir = os.path.join(DATASET_ROOT, class_name)
    output_dir = os.path.join(SAVE_ROOT, class_name)
    os.makedirs(output_dir, exist_ok=True)

    videos = [v for v in os.listdir(input_dir) if v.endswith(".mp4")]
    skipped = 0

    for video in tqdm(videos, desc=f"Precomputing temporal faces [{class_name}]"):
        save_name = video.replace(".mp4", ".npy")
        save_path = os.path.join(output_dir, save_name)

        if os.path.exists(save_path):
            continue  # already cached from a previous run — resumable

        video_path = os.path.join(input_dir, video)

        try:
            frames = sample_frames(video_path, num_frames=num_frames, size=(224, 224))

            face_sequence = np.zeros((num_frames, 224, 224, 3), dtype=np.uint8)
            for i in range(frames.shape[0]):
                face_sequence[i] = detect_face(frames[i])

            np.save(save_path, face_sequence)

        except Exception as e:
            # A single corrupt/unreadable video shouldn't kill the whole run.
            skipped += 1
            print(f"  [skip] {video}: {e}")

    if skipped:
        print(f"[{class_name}] Skipped {skipped}/{len(videos)} unreadable videos.")


if __name__ == "__main__":
    process_class("real")
    process_class("fake")
    print(f"\nTemporal face sequences cached to: {SAVE_ROOT}")
