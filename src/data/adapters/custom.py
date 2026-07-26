import os

from src.data.base_dataset import BaseDatasetAdapter, DatasetUnavailableError, VideoRecord, probe_video

SETUP_NOTE = """
The "custom" dataset has no download step — you provide the videos
yourself. Just place them here (no code changes needed):

  data/raw/custom/
  ├── real/*.mp4
  └── fake/*.mp4

Then run:
  python -m src.data.metadata      (regenerates data/metadata/metadata.csv,
                                     picking up your custom videos automatically)
  python -m src.preprocessing.create_splits

Identity fallback: custom videos have no identity information by default,
so each video is treated as its own identity group (equivalent to a plain
video-level split for this dataset specifically — see create_splits.py).
If your custom videos DO have known identities/subjects and you want
identity-aware splitting for them too, name files like
"<identity>__<anything>.mp4" (double underscore separator) and this adapter
will pick up the identity prefix automatically.
"""


class CustomAdapter(BaseDatasetAdapter):
    name = "custom"

    def download(self):
        if not os.path.isdir(self.dataset_root):
            raise DatasetUnavailableError(SETUP_NOTE)
        # Nothing to fetch — videos are user-provided. This just verifies
        # the expected folders exist so failures are caught early and
        # clearly rather than silently yielding zero videos later.
        for class_name in ("real", "fake"):
            class_dir = os.path.join(self.dataset_root, class_name)
            if not os.path.isdir(class_dir):
                raise DatasetUnavailableError(
                    f"Expected folder not found: {class_dir}\n" + SETUP_NOTE
                )

    def list_videos(self):
        for class_name, label in (("real", 0), ("fake", 1)):
            class_dir = os.path.join(self.dataset_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.endswith(".mp4"):
                    continue

                identity = None
                if "__" in fname:
                    identity = fname.split("__", 1)[0]

                rel_path = os.path.join(class_name, fname)
                abs_path = os.path.join(class_dir, fname)
                probed = probe_video(abs_path)

                yield VideoRecord(
                    dataset_name=self.name,
                    video_path=rel_path,
                    label=label,
                    identity=identity,
                    **probed,
                )
