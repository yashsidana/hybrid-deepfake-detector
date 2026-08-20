import os
import re

import kagglehub

from src.data.base_dataset import BaseDatasetAdapter, VideoRecord, probe_video

# Google/Jigsaw's Deep Fake Detection (DFD) dataset, contributed to the
# FaceForensics++ ecosystem and released openly for research use. Unlike
# DFDC (Kaggle competition, gated) or FaceForensics++'s own manipulated
# sequences (TU Munich manual-approval form), this specific portion is
# mirrored as a plain public Kaggle DATASET (not a competition) with no
# accept-rules gate -- kagglehub.dataset_download() works the same way it
# does for Celeb-DF v2, no manual step required.
#
# Filename convention (same source/target identity-pairing idea as
# FaceForensics++, just double-underscore-separated):
#   Original:    {actor_id}__{action}.mp4              e.g. 01__exit_phone_room.mp4
#   Manipulated: {source_id}_{target_id}__{action}_{hash}.mp4
#                                                        e.g. 01_02__exit_phone_room_XXXXXXXX.mp4
_MANIPULATED_ID_PATTERN = re.compile(r"^(\d+)_(\d+)__")
_ORIGINAL_ID_PATTERN = re.compile(r"^(\d+)__")


def _extract_identity(filename):
    """
    Returns a comma-joined string of identity tokens (namespaced with
    "dfd_" so they can never collide with another dataset's identity
    tokens, e.g. Celeb-DF's "id0" scheme), or None if the filename doesn't
    match either known pattern.
    """
    m = _MANIPULATED_ID_PATTERN.match(filename)
    if m:
        source, target = m.group(1), m.group(2)
        tokens = sorted({f"dfd_{source}", f"dfd_{target}"})
        return ",".join(tokens)

    m = _ORIGINAL_ID_PATTERN.match(filename)
    if m:
        return f"dfd_{m.group(1)}"

    return None


class DFDAdapter(BaseDatasetAdapter):
    """
    Fully automated, like CelebDFAdapter -- no manual approval gate. See
    module docstring above for why this dataset (specifically) is exempt
    from the DFDC/FaceForensics++/FakeAVCeleb manual-approval problem.
    """

    name = "dfd"

    KAGGLE_HANDLE = "sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset"

    def download(self, max_retries=10, retry_delay=5):
        import time
        path = None
        for attempt in range(1, max_retries + 1):
            try:
                print(f"[DFDAdapter] Downloading / verifying dataset (attempt {attempt}/{max_retries})...")
                path = kagglehub.dataset_download(self.KAGGLE_HANDLE)
                break
            except Exception as e:
                print(f"[DFDAdapter] Download interrupted: {e}")
                if attempt < max_retries:
                    print(f"[DFDAdapter] Retrying in {retry_delay}s (will resume from cached byte offset)...")
                    time.sleep(retry_delay)
                else:
                    raise

        os.makedirs(self.dataset_root, exist_ok=True)

        import shutil
        for dirpath, _, filenames in os.walk(path):
            for fname in filenames:
                if not fname.endswith(".mp4"):
                    continue
                src = os.path.join(dirpath, fname)
                dst = os.path.join(self.dataset_root, fname)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    def list_videos(self):
        """
        Flat scan for .mp4 files anywhere under dataset_root, classified
        by filename pattern rather than a hardcoded folder depth -- the
        upstream Kaggle dataset nests DFD_manipulated_sequences one level
        deeper than DFD_original sequences (an inconsistency in the
        original upload, not something this adapter should assume is
        stable), and download() above already flattens everything into
        dataset_root directly, matching the convention custom.py/celebdf.py
        use elsewhere in this project.
        """
        if not os.path.isdir(self.dataset_root):
            return

        for fname in sorted(os.listdir(self.dataset_root)):
            if not fname.endswith(".mp4"):
                continue

            abs_path = os.path.join(self.dataset_root, fname)
            identity = _extract_identity(fname)
            # Manipulated filenames have TWO leading numeric tokens
            # separated by "_" before "__"; originals have exactly one.
            label = 1 if _MANIPULATED_ID_PATTERN.match(fname) else 0

            probed = probe_video(abs_path)

            yield VideoRecord(
                dataset_name=self.name,
                video_path=fname,
                label=label,
                identity=identity,
                **probed,
            )
