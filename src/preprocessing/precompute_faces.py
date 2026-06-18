import os
import cv2
from tqdm import tqdm

from src.preprocessing.frame_sampler import sample_frames
from src.preprocessing.face_detection import detect_face

DATASET_ROOT = "data/raw/celebdf"
SAVE_ROOT = "data/processed/semantic_faces"


def process_class(class_name):
    input_dir = os.path.join(DATASET_ROOT, class_name)
    output_dir = os.path.join(SAVE_ROOT, class_name)

    os.makedirs(output_dir, exist_ok=True)

    for video in tqdm(os.listdir(input_dir)):
        if not video.endswith(".mp4"):
            continue

        video_path = os.path.join(input_dir, video)

        # Only sample ONE key frame
        frames = sample_frames(
            video_path,
            num_frames=1
        )

        frame = frames[0]

        face = detect_face(frame)

        save_name = video.replace(".mp4", ".jpg")

        cv2.imwrite(
            os.path.join(output_dir, save_name),
            face
        )


if __name__ == "__main__":
    process_class("real")
    process_class("fake")