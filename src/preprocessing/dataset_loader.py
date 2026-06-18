import os
import pandas as pd
import torch

from torch.utils.data import Dataset

from src.preprocessing.frame_sampler import sample_frames
from src.preprocessing.face_detection import detect_face
from src.preprocessing.temporal_loader import create_temporal_pairs

class CelebDFDataset(Dataset):
    def __init__(self, csv_path, dataset_root="data/raw/celebdf"):
        self.df = pd.read_csv(csv_path)
        self.dataset_root = dataset_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Full video path
        video_path = os.path.join(self.dataset_root, row["video"])

        # Label
        label = torch.tensor(row["label"], dtype=torch.long)

        # Sample frames
        frames = sample_frames(
            video_path,
            num_frames=16,
            size=(224, 224)
        )

        # Face detection for semantic branch
        face_frames = []

        for frame in frames:
            face = detect_face(frame)

            face = torch.tensor(face).permute(2, 0, 1).float() / 255.0
            face_frames.append(face)

        face_frames = torch.stack(face_frames)

        # Temporal pairs for temporal branch
        temporal_pairs = create_temporal_pairs(frames)

        return {
            "frames": frames,                   # raw frames
            "face_frames": face_frames,        # semantic input
            "temporal_pairs": temporal_pairs,  # temporal input
            "label": label
        }