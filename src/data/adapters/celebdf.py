import os
import re

import kagglehub

from src.data.base_dataset import BaseDatasetAdapter, VideoRecord, probe_video

# Celeb-DF v2 filename conventions (identities are encoded in the filename
# even after setup_dataset.py flattens the folder structure):
#   Celeb-real:        id{N}_{M}.mp4          e.g. id0_0000.mp4
#   YouTube-real:       {M}.mp4                e.g. 00170.mp4 (no identity encoded)
#   Celeb-synthesis:    id{N}_id{K}_{M}.mp4    e.g. id0_id1_0003.mp4 (source_target)
_ID_PATTERN = re.compile(r"id(\d+)")


def _extract_identity_tokens(filename):
    """
    Returns a comma-joined string of identity tokens found in the filename
    (e.g. "id0,id1"), or None if the filename encodes no identity (e.g.
    YouTube-real clips). create_splits.py groups videos that share ANY
    token into the same split — this matters because a fake video's source
    AND target identity both need to stay on the same side of the
    train/val/test boundary as any other video referencing either identity.
    """
    tokens = _ID_PATTERN.findall(filename)
    if not tokens:
        return None
    return ",".join(f"id{t}" for t in sorted(set(tokens), key=int))


class CelebDFAdapter(BaseDatasetAdapter):
    name = "celebdf"

    KAGGLE_HANDLE = "reubensuju/celeb-df-v2"

    def download(self):
        """
        Fully automated — no licensing gate. Mirrors the original
        setup_dataset.py behavior exactly, so existing local data (already
        downloaded under the old, pre-multi-dataset pipeline) is untouched;
        this just re-expresses the same download as an adapter method.
        """
        path = kagglehub.dataset_download(self.KAGGLE_HANDLE)

        for class_name in ("real", "fake"):
            dst_dir = os.path.join(self.dataset_root, class_name)
            os.makedirs(dst_dir, exist_ok=True)

        import shutil
        for dirpath, _, filenames in os.walk(path):
            for fname in filenames:
                if not fname.endswith(".mp4"):
                    continue
                lower_path = dirpath.lower()
                if "real" in lower_path:
                    class_name = "real"
                elif "fake" in lower_path or "synthesis" in lower_path:
                    class_name = "fake"
                else:
                    continue
                src = os.path.join(dirpath, fname)
                dst = os.path.join(self.dataset_root, class_name, fname)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    def list_videos(self):
        for class_name, label in (("real", 0), ("fake", 1)):
            class_dir = os.path.join(self.dataset_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.endswith(".mp4"):
                    continue
                rel_path = os.path.join(class_name, fname)
                abs_path = os.path.join(class_dir, fname)
                probed = probe_video(abs_path)

                yield VideoRecord(
                    dataset_name=self.name,
                    video_path=rel_path,
                    label=label,
                    identity=_extract_identity_tokens(fname),
                    **probed,
                )
