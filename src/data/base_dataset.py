"""
Base interface every dataset adapter implements.

Design goal: adding a 7th dataset later should require writing ONE new
adapter file (subclassing BaseDatasetAdapter) and registering it in
src/data/registry.py — nothing else in the codebase should need to change.
Everything downstream (metadata generation, splitting, semantic/temporal
preprocessing) talks to datasets only through this interface.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import cv2


@dataclass
class VideoRecord:
    """
    One row of the project-wide metadata table (data/metadata/metadata.csv).
    This is the single source of truth every downstream module reads from —
    no module should ever construct a raw video path by hand outside of an
    adapter's list_videos().
    """

    dataset_name: str
    video_path: str          # relative to data/raw/<dataset_name>/
    label: int                # 0 = real, 1 = fake
    identity: Optional[str] = None   # None -> no identity info available (see fallback strategy in create_splits.py)
    split: Optional[str] = None      # filled in later by create_splits.py; None at metadata-generation time
    fps: Optional[float] = None
    duration: Optional[float] = None
    resolution: Optional[str] = None   # "WIDTHxHEIGHT"
    codec: Optional[str] = None
    compression_level: Optional[str] = None  # e.g. Celeb-DF/FF++ don't always expose this; None if unknown
    number_of_frames: Optional[int] = None
    face_count: Optional[int] = None   # optional, left None unless a caller explicitly probes for it (expensive)

    def as_dict(self):
        return {
            "dataset": self.dataset_name,
            "video_path": self.video_path,
            "label": self.label,
            "identity": self.identity,
            "split": self.split,
            "fps": self.fps,
            "duration": self.duration,
            "resolution": self.resolution,
            "codec": self.codec,
            "compression_level": self.compression_level,
            "number_of_frames": self.number_of_frames,
            "face_count": self.face_count,
        }


def probe_video(path):
    """
    Shared helper: reads fps/duration/resolution/frame-count via OpenCV.
    Used by every adapter so probing logic (and its failure handling) lives
    in exactly one place. Returns a dict of Nones if the file can't be
    opened — callers should not treat that as fatal, just incomplete
    metadata (surfaced later by validate.py).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return {"fps": None, "duration": None, "resolution": None, "number_of_frames": None}

    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    duration = (frame_count / fps) if (frame_count and fps) else None
    resolution = f"{width}x{height}" if (width and height) else None

    cap.release()

    return {
        "fps": fps,
        "duration": duration,
        "resolution": resolution,
        "number_of_frames": frame_count,
    }


class DatasetUnavailableError(Exception):
    """
    Raised by download() for datasets that cannot be automatically
    downloaded (licensing/authentication/manual-approval gates). The message
    always points to the official, documented manual process — never a
    workaround.
    """


class BaseDatasetAdapter(ABC):
    """
    Subclass this for every supported dataset. See adapters/celebdf.py for
    the simplest fully-automated example, adapters/faceforensics.py for the
    manual-only pattern, and adapters/custom.py for the zero-download
    user-provided-folder pattern.
    """

    name: str = "base"

    def __init__(self, raw_root="data/raw"):
        self.raw_root = raw_root
        self.dataset_root = os.path.join(raw_root, self.name)

    def is_available(self) -> bool:
        """True if this dataset's raw videos already exist locally."""
        return os.path.isdir(self.dataset_root) and any(
            self._walk_video_files()
        )

    @abstractmethod
    def download(self) -> None:
        """
        Fetch (or, for manual-only datasets, raise DatasetUnavailableError
        with instructions for) the raw dataset into self.dataset_root.
        """
        raise NotImplementedError

    @abstractmethod
    def list_videos(self) -> Iterator[VideoRecord]:
        """Yield one VideoRecord per video found in self.dataset_root."""
        raise NotImplementedError

    def _walk_video_files(self, extensions=(".mp4", ".avi", ".mov", ".mkv")):
        if not os.path.isdir(self.dataset_root):
            return
        for dirpath, _, filenames in os.walk(self.dataset_root):
            for fname in filenames:
                if fname.lower().endswith(extensions):
                    yield os.path.join(dirpath, fname)
