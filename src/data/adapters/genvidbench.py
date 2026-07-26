import os

from src.data.base_dataset import BaseDatasetAdapter, DatasetUnavailableError, VideoRecord, probe_video

SETUP_NOTE = """
GenVidBench IS publicly downloadable (no access-request gate, unlike
FaceForensics++/FakeAVCeleb) — it's hosted on HuggingFace. However:

  1. I could not verify a single, stable, authoritative HuggingFace repo ID
     for it at the time this adapter was written — multiple mirrors/subset
     uploads exist under different namespaces, and the official project
     page (https://genvidbench.github.io/) is the right place to confirm
     the current canonical link before you download anything.
  2. The full released benchmark is ~6.78 MILLION videos. Downloading all
     of it is almost certainly not what you want for a capstone project —
     pick a small subset.
  3. Some source videos (Pika, VideoCraftV2, ModelScope, T2V-Zero) are only
     obtainable from a separate service (VidProM), not bundled with the
     main HuggingFace release.

Because of point 1, this adapter requires you to explicitly set the
HuggingFace repo ID in config.yaml (datasets.genvidbench.hf_repo_id) rather
than guessing one — this is safer than silently pointing at a possibly
wrong or unofficial mirror.

Once you've confirmed the repo and downloaded a subset, reorganize it into
this project's standard flat layout (same convention as celebdf/custom):

  data/raw/genvidbench/
  ├── real/*.mp4
  └── fake/*.mp4
"""


class GenVidBenchAdapter(BaseDatasetAdapter):
    name = "genvidbench"

    def __init__(self, raw_root="data/raw", hf_repo_id=None, allow_patterns=None):
        super().__init__(raw_root)
        self.hf_repo_id = hf_repo_id
        self.allow_patterns = allow_patterns

    def download(self):
        if not self.hf_repo_id:
            raise DatasetUnavailableError(SETUP_NOTE)

        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise DatasetUnavailableError(
                "huggingface_hub is required for GenVidBench. "
                "Install it with: pip install huggingface_hub\n\n" + SETUP_NOTE
            ) from e

        os.makedirs(self.dataset_root, exist_ok=True)

        # allow_patterns lets you fetch a small slice instead of the full
        # 6.78M-video release — strongly recommended. Configure via
        # config.yaml's datasets.genvidbench.allow_patterns.
        snapshot_download(
            repo_id=self.hf_repo_id,
            repo_type="dataset",
            local_dir=self.dataset_root,
            allow_patterns=self.allow_patterns,
        )

        print(
            "Downloaded from HuggingFace. NOTE: you likely need to manually "
            "reorganize the downloaded files into data/raw/genvidbench/"
            "{real,fake}/*.mp4 — the HF repo's internal layout varies and "
            "isn't something this adapter can assume. See SETUP_NOTE in "
            "src/data/adapters/genvidbench.py."
        )

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
                    identity=None,  # no persistent identity concept for generated clips
                    **probed,
                )
