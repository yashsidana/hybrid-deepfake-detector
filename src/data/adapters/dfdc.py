import json
import os

import kagglehub

from src.data.base_dataset import BaseDatasetAdapter, DatasetUnavailableError, VideoRecord, probe_video

COMPETITION_HANDLE = "deepfake-detection-challenge"
RULES_URL = f"https://www.kaggle.com/c/{COMPETITION_HANDLE}/rules"


class DFDCAdapter(BaseDatasetAdapter):
    """
    DFDC is hosted as a Kaggle COMPETITION, not a public dataset — Kaggle
    requires every user to manually click "I Understand and Accept" on the
    competition rules page before any download (API or otherwise) is
    permitted for that account. This is a one-time, per-account gate that
    cannot and should not be scripted around.

    Once accepted, kagglehub.competition_download() works the same way
    kagglehub.dataset_download() does for Celeb-DF.

    The full training set is ~470GB across 50 zip parts
    (dfdc_train_part_0 .. dfdc_train_part_49). Downloading everything is
    rarely practical for a capstone project — download() here fetches a
    single part by default; pass part_index to change which one, or call it
    in a loop for more.
    """

    name = "dfdc"

    def download(self, part_index=0):
        try:
            path = kagglehub.competition_download(
                COMPETITION_HANDLE,
                path=("train_sample_videos.zip" if part_index == "sample" else f"dfdc_train_part_{part_index}.zip"),
            )
        except Exception as e:
            raise DatasetUnavailableError(
                "DFDC download failed. This almost always means the "
                "competition rules haven't been accepted yet for this "
                "Kaggle account (a manual, one-time step Kaggle requires — "
                "not something this code can do for you).\n\n"
                f"Fix: log into Kaggle in a browser, visit {RULES_URL}, "
                "and click 'I Understand and Accept'. Then re-run this.\n\n"
                f"Original error: {e}"
            ) from e

        os.makedirs(self.dataset_root, exist_ok=True)

        import shutil
        import zipfile

        if os.path.isfile(path) and path.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                zf.extractall(self.dataset_root)
        elif os.path.isdir(path):
            for item in os.listdir(path):
                src = os.path.join(path, item)
                dst = os.path.join(self.dataset_root, item)
                if not os.path.exists(dst):
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy(src, dst)

    def list_videos(self):
        """
        Scans every dfdc_train_part_* (or train_sample_videos) folder found
        under data/raw/dfdc/, reading each part's own metadata.json for
        label and the "original" field (the real video a fake was derived
        from — used as the identity-grouping key, since DFDC doesn't
        provide named identities directly).
        """
        if not os.path.isdir(self.dataset_root):
            return

        for entry in sorted(os.listdir(self.dataset_root)):
            part_dir = os.path.join(self.dataset_root, entry)
            if not os.path.isdir(part_dir):
                continue

            metadata_path = os.path.join(part_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                continue  # not a recognized DFDC part folder — skip rather than guess

            with open(metadata_path) as f:
                part_metadata = json.load(f)

            for fname, info in part_metadata.items():
                video_abs_path = os.path.join(part_dir, fname)
                if not os.path.exists(video_abs_path):
                    continue  # metadata references a video not actually present locally

                label = 1 if info.get("label", "").upper() == "FAKE" else 0
                # Group a fake with the real video it was generated from —
                # this is DFDC's own documented identity-adjacent signal.
                identity = info.get("original", fname)

                rel_path = os.path.join(entry, fname)
                probed = probe_video(video_abs_path)

                yield VideoRecord(
                    dataset_name=self.name,
                    video_path=rel_path,
                    label=label,
                    identity=identity,
                    **probed,
                )
