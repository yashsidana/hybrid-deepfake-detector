import os

from src.data.base_dataset import BaseDatasetAdapter, DatasetUnavailableError, VideoRecord, probe_video

ACCESS_FORM_URL = (
    "https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform"
)

MANUAL_SETUP_INSTRUCTIONS = f"""
FaceForensics++ cannot be downloaded automatically — there is no public API.
The dataset owners (TU Munich) require manual approval for every request:

  1. Fill out the official access form: {ACCESS_FORM_URL}
  2. Wait for manual review (their README says allow up to a week).
  3. You'll receive an email with a personal download-script link and a
     one-time-use download-FaceForensics.py script.
  4. Run their script yourself, e.g.:
       python download-FaceForensics.py <output_dir> -d all -c c23 -t videos
  5. Point this project at the result — see below for the expected layout.

Expected layout once you have the data (place it under, or symlink to,
data/raw/faceforensics/):

  data/raw/faceforensics/
  ├── original_sequences/youtube/c23/videos/*.mp4
  └── manipulated_sequences/
      ├── Deepfakes/c23/videos/*.mp4
      ├── Face2Face/c23/videos/*.mp4
      ├── FaceSwap/c23/videos/*.mp4
      └── NeuralTextures/c23/videos/*.mp4

This is their own documented directory structure — nothing here renames or
reorganizes it, by design (preserving original hierarchy per project
convention).
"""


class FaceForensicsAdapter(BaseDatasetAdapter):
    """
    Manual-only dataset — see MANUAL_SETUP_INSTRUCTIONS. This adapter's job
    is entirely to READ an already-downloaded copy correctly (identity
    extraction, compression-level tagging from the c23/c40/raw folder name),
    never to fetch it.
    """

    name = "faceforensics"

    MANIPULATION_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
    COMPRESSION_FOLDERS = ("raw", "c23", "c40")

    def download(self):
        raise DatasetUnavailableError(MANUAL_SETUP_INSTRUCTIONS)

    def list_videos(self):
        # Original (real) sequences
        for compression in self.COMPRESSION_FOLDERS:
            real_dir = os.path.join(self.dataset_root, "original_sequences", "youtube", compression, "videos")
            if not os.path.isdir(real_dir):
                continue
            for fname in sorted(os.listdir(real_dir)):
                if not fname.endswith(".mp4"):
                    continue
                stem = fname.replace(".mp4", "")
                abs_path = os.path.join(real_dir, fname)
                rel_path = os.path.join("original_sequences", "youtube", compression, "videos", fname)
                probed = probe_video(abs_path)

                yield VideoRecord(
                    dataset_name=self.name,
                    video_path=rel_path,
                    label=0,
                    identity=stem,
                    compression_level=compression,
                    **probed,
                )

        # Manipulated (fake) sequences
        for method in self.MANIPULATION_METHODS:
            for compression in self.COMPRESSION_FOLDERS:
                fake_dir = os.path.join(self.dataset_root, "manipulated_sequences", method, compression, "videos")
                if not os.path.isdir(fake_dir):
                    continue
                for fname in sorted(os.listdir(fake_dir)):
                    if not fname.endswith(".mp4"):
                        continue
                    stem = fname.replace(".mp4", "")
                    # FF++ manipulated filenames are "{source}_{target}.mp4"
                    parts = stem.split("_")
                    identity = ",".join(sorted(set(parts))) if len(parts) == 2 else stem

                    abs_path = os.path.join(fake_dir, fname)
                    rel_path = os.path.join("manipulated_sequences", method, compression, "videos", fname)
                    probed = probe_video(abs_path)

                    yield VideoRecord(
                        dataset_name=self.name,
                        video_path=rel_path,
                        label=1,
                        identity=identity,
                        compression_level=compression,
                        **probed,
                    )
