import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Windows' spawn-based multiprocessing re-imports torch/torchvision from
# scratch in every worker process, which is slow to start and fragile to
# interrupt (Ctrl+C during worker spawn can leave the process looking
# "stuck"). Default to 0 workers there; Linux/Colab can fork cheaply, so
# keep parallel loading on those platforms. Always overridable explicitly.
_DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 2

# Root where precompute_temporal.py writes cached face sequences:
#   data/processed/temporal/<dataset>/<video_path with .mp4 -> .npy>
SEQUENCES_ROOT = "data/processed/temporal"

# Same splits directory used by image_loader.py — no new splitting logic.
SPLITS_ROOT = "data/splits"

# ImageNet normalization, since the per-frame CNN backbone (ResNet-18) is
# ImageNet-pretrained.
_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class TemporalSequenceDataset(Dataset):
    """
    Loads precomputed 16-frame face sequences using the same video-level
    splits (train.csv/val.csv/test.csv) as the semantic branch's
    FaceCSVDataset. This guarantees both branches train/validate/test on the
    exact same videos, which Phase 3 fusion depends on.
    """

    def __init__(self, csv_path, sequences_root=SEQUENCES_ROOT):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Split file not found: {csv_path}. "
                f"Run `python -m src.preprocessing.create_splits` first."
            )

        df = pd.read_csv(csv_path)
        self.sequences_root = sequences_root

        self.samples = []
        missing = 0

        for _, row in df.iterrows():
            seq_path = self._resolve_sequence_path(row["dataset"], row["video_path"])
            if os.path.exists(seq_path):
                self.samples.append((seq_path, int(row["label"])))
            else:
                missing += 1

        if missing:
            print(
                f"[TemporalSequenceDataset] Warning: {missing} row(s) in "
                f"{os.path.basename(csv_path)} have no precomputed sequence "
                f"and were skipped. Run precompute_temporal.py if this "
                f"number is large."
            )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No usable samples found for {csv_path}. Did you run "
                f"precompute_temporal.py after create_splits.py?"
            )

    def _resolve_sequence_path(self, dataset, video_rel_path):
        stem = video_rel_path.replace(".mp4", ".npy")
        return os.path.join(self.sequences_root, dataset, stem)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, label = self.samples[idx]

        # [T, 224, 224, 3] uint8, BGR (same contract as the semantic branch)
        sequence = np.load(seq_path)

        # -> [T, 3, 224, 224] float in [0, 1], RGB
        sequence = sequence[..., ::-1].copy()  # BGR -> RGB
        sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2).float() / 255.0

        # Normalize each frame with ImageNet stats
        sequence = torch.stack([_normalize(frame) for frame in sequence])

        return sequence, torch.tensor(label, dtype=torch.long)


def get_dataloaders(batch_size=8, num_workers=_DEFAULT_NUM_WORKERS, splits_root=SPLITS_ROOT):
    """
    Returns (train_loader, val_loader, test_loader) for the temporal branch,
    backed by the same CSVs as the semantic branch's get_dataloaders().

    Default batch_size is smaller than the semantic branch's (8 vs 16) since
    each sample here is a 16-frame sequence rather than a single image —
    16x the tensor volume per sample.
    """
    train_set = TemporalSequenceDataset(os.path.join(splits_root, "train.csv"))
    val_set = TemporalSequenceDataset(os.path.join(splits_root, "val.csv"))
    test_set = TemporalSequenceDataset(os.path.join(splits_root, "test.csv"))

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
