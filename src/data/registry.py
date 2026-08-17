"""
Single place that knows about every dataset adapter that exists. Adding a
7th dataset later: write the adapter class, add ONE line here. Nothing
else in the codebase needs to change.
"""

from src.data.adapters.celebdf import CelebDFAdapter
from src.data.adapters.custom import CustomAdapter
from src.data.adapters.dfd import DFDAdapter
from src.data.adapters.dfdc import DFDCAdapter
from src.data.adapters.faceforensics import FaceForensicsAdapter
from src.data.adapters.fakeavceleb import FakeAVCelebAdapter
from src.data.adapters.genvidbench import GenVidBenchAdapter

ADAPTER_REGISTRY = {
    "celebdf": CelebDFAdapter,
    "dfd": DFDAdapter,
    "dfdc": DFDCAdapter,
    "faceforensics": FaceForensicsAdapter,
    "fakeavceleb": FakeAVCelebAdapter,
    "genvidbench": GenVidBenchAdapter,
    "custom": CustomAdapter,
}


def get_enabled_adapters(config, raw_root="data/raw"):
    """
    config is the parsed config.yaml dict. Reads config["datasets"]
    (name -> bool) and instantiates only the enabled adapters.

    Datasets absent from config["datasets"] default to disabled — an
    adapter must be explicitly turned on, never silently active.
    """
    datasets_cfg = config.get("datasets", {})
    enabled = []

    for name, adapter_cls in ADAPTER_REGISTRY.items():
        entry = datasets_cfg.get(name, False)

        # Support both `celebdf: true` (bool) and
        # `genvidbench: {enabled: true, hf_repo_id: "..."}`  (dict) forms —
        # the latter is used by adapters needing extra config, like
        # GenVidBench's repo ID.
        if isinstance(entry, dict):
            is_enabled = entry.get("enabled", False)
            extra_kwargs = {k: v for k, v in entry.items() if k != "enabled"}
        else:
            is_enabled = bool(entry)
            extra_kwargs = {}

        if not is_enabled:
            continue

        enabled.append(adapter_cls(raw_root=raw_root, **extra_kwargs))

    return enabled
