import os
import warnings

from src.data.base_dataset import BaseDatasetAdapter, DatasetUnavailableError, VideoRecord, probe_video

ACCESS_FORM_URL = "https://bit.ly/38prlVO"

MANUAL_SETUP_INSTRUCTIONS = f"""
FakeAVCeleb cannot be downloaded automatically — the DASH Lab (Sungkyunkwan
University) requires manual approval for every request:

  1. Fill out the official request form: {ACCESS_FORM_URL}
  2. Wait for manual screening of your Data Use Agreement (their README
     notes this is a manual process specifically to limit misuse).
  3. You'll receive an email with a Google Drive link and download script.
  4. Download and extract it yourself.
  5. Point this project at the result — see below for the expected layout.

Expected layout once you have the data (place it under, or symlink to,
data/raw/fakeavceleb/):

  data/raw/fakeavceleb/
  ├── RealVideo-RealAudio/<race>/<gender>/id<N>/*.mp4
  ├── FakeVideo-FakeAudio/<race>/<gender>/id<N>/<method>/*.mp4
  ├── FakeVideo-RealAudio/<race>/<gender>/id<N>/<method>/*.mp4
  └── RealVideo-FakeAudio/<race>/<gender>/id<N>/<method>/*.mp4

Note: this project's current pipeline is video-only (no audio branch yet),
so the label used here reflects VIDEO manipulation status only — e.g.
"RealVideo-FakeAudio" is labeled REAL, since the visual content wasn't
manipulated, even though the audio was. Revisit this if/when an audio
branch is added.
"""


class FakeAVCelebAdapter(BaseDatasetAdapter):
    name = "fakeavceleb"

    CATEGORY_LABELS = {
        "RealVideo-RealAudio": 0,
        "RealVideo-FakeAudio": 0,
        "FakeVideo-RealAudio": 1,
        "FakeVideo-FakeAudio": 1,
    }

    def download(self):
        raise DatasetUnavailableError(MANUAL_SETUP_INSTRUCTIONS)

    def list_videos(self):
        found_expected_structure = any(
            os.path.isdir(os.path.join(self.dataset_root, category))
            for category in self.CATEGORY_LABELS
        )

        if not found_expected_structure:
            if os.path.isdir(self.dataset_root):
                warnings.warn(
                    "FakeAVCeleb: none of the expected category folders "
                    f"({list(self.CATEGORY_LABELS)}) were found under "
                    f"{self.dataset_root}. If your copy uses a different "
                    "layout (this can vary by download batch), adjust "
                    "list_videos() in src/data/adapters/fakeavceleb.py to "
                    "match your actual folder structure."
                )
            return

        for category, label in self.CATEGORY_LABELS.items():
            category_dir = os.path.join(self.dataset_root, category)
            if not os.path.isdir(category_dir):
                continue

            for dirpath, _, filenames in os.walk(category_dir):
                for fname in filenames:
                    if not fname.endswith(".mp4"):
                        continue

                    abs_path = os.path.join(dirpath, fname)
                    rel_path = os.path.relpath(abs_path, self.dataset_root)

                    # Identity = the id<N> path segment, if present
                    identity = None
                    for part in rel_path.split(os.sep):
                        if part.startswith("id") and part[2:].isdigit():
                            identity = part
                            break

                    probed = probe_video(abs_path)

                    yield VideoRecord(
                        dataset_name=self.name,
                        video_path=rel_path,
                        label=label,
                        identity=identity,
                        **probed,
                    )
