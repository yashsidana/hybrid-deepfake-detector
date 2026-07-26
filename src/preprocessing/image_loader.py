import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Windows' spawn-based multiprocessing re-imports torch/torchvision from
# scratch in every worker process, which is slow to start and fragile to
# interrupt (Ctrl+C during worker spawn can leave the process looking
# "stuck"). Default to 0 workers there; Linux/Colab can fork cheaply, so
# keep parallel loading on those platforms. Always overridable explicitly.
_DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 2

# Root where precompute_faces.py writes cropped face images:
#   data/processed/semantic_faces/real/<video_stem>.jpg
#   data/processed/semantic_faces/fake/<video_stem>.jpg
FACES_ROOT = "data/processed/semantic_faces"

# Root where create_splits.py writes train.csv / val.csv / test.csv
SPLITS_ROOT = "data/splits"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


class FaceCSVDataset(Dataset):
    """
    Loads precomputed face images using the exact video-level splits produced
    by create_splits.py, instead of re-splitting on disk.

    This replaces the old behavior where image_loader.py called
    torchvision.datasets.ImageFolder + random_split on the whole
    semantic_faces/ folder, silently discarding train.csv/val.csv/test.csv
    and re-shuffling train/test membership on every run.

    Each CSV row is: video (e.g. "real/00001.mp4"), label (0=real, 1=fake).
    We resolve that to the corresponding precomputed face image path.
    """

    def __init__(self, csv_path, faces_root=FACES_ROOT, transform=transform):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Split file not found: {csv_path}. "
                f"Run `python -m src.preprocessing.create_splits` first."
            )

        df = pd.read_csv(csv_path)
        self.faces_root = faces_root
        self.transform = transform

        self.samples = []
        missing = 0

        for _, row in df.iterrows():
            face_path = self._resolve_face_path(row["video"])
            if os.path.exists(face_path):
                self.samples.append((face_path, int(row["label"])))
            else:
                missing += 1

        if missing:
            print(
                f"[FaceCSVDataset] Warning: {missing} row(s) in "
                f"{os.path.basename(csv_path)} have no precomputed face image "
                f"and were skipped. If this number is large, run "
                f"precompute_faces.py again — it may not have finished, or "
                f"face detection failed on those videos."
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No usable samples found for {csv_path}. Did you run "
                f"precompute_faces.py after create_splits.py?"
            )

    def _resolve_face_path(self, video_rel_path):
        # "real/00001.mp4" -> "data/processed/semantic_faces/real/00001.jpg"
        stem = video_rel_path.replace(".mp4", ".jpg")
        return os.path.join(self.faces_root, stem)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_path, label = self.samples[idx]
        image = Image.open(face_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size=16, num_workers=_DEFAULT_NUM_WORKERS, splits_root=SPLITS_ROOT):
    """
    Returns (train_loader, val_loader, test_loader), each backed by the
    corresponding CSV from create_splits.py. No random re-splitting —
    the same three CSVs are used everywhere in the pipeline, including by
    dataset_loader.py's CelebDFDataset for the temporal branch.
    """
    train_set = FaceCSVDataset(os.path.join(splits_root, "train.csv"))
    val_set = FaceCSVDataset(os.path.join(splits_root, "val.csv"))
    test_set = FaceCSVDataset(os.path.join(splits_root, "test.csv"))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
