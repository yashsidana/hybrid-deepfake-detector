"""
Generates data/metadata/metadata.csv by calling list_videos() on every
dataset enabled in config.yaml. This file becomes the single source of
truth every downstream module (create_splits.py, precompute_faces.py,
precompute_temporal.py) reads from — none of them should ever list a raw
dataset folder directly.
"""

import os

import pandas as pd
import yaml

from src.data.registry import get_enabled_adapters

CONFIG_PATH = "config/config.yaml"
METADATA_PATH = "data/metadata/metadata.csv"


def load_config(config_path=CONFIG_PATH):
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_metadata(config=None, output_path=METADATA_PATH):
    if config is None:
        config = load_config()

    adapters = get_enabled_adapters(config)

    if not adapters:
        raise RuntimeError(
            "No datasets are enabled in config.yaml's `datasets` section. "
            "Enable at least one (e.g. `celebdf: true`) before generating metadata."
        )

    all_records = []

    for adapter in adapters:
        if not adapter.is_available():
            print(
                f"[{adapter.name}] No raw videos found at "
                f"{adapter.dataset_root} — skipping. Run its download step "
                f"first if you want it included."
            )
            continue

        print(f"[{adapter.name}] Scanning videos...")
        count = 0
        for record in adapter.list_videos():
            all_records.append(record.as_dict())
            count += 1
        print(f"[{adapter.name}] {count} videos found.")

    if not all_records:
        raise RuntimeError(
            "No videos found across any enabled dataset. Check that "
            "download steps have actually been run and raw videos exist "
            "under data/raw/<dataset>/."
        )

    df = pd.DataFrame(all_records)

    # Duplicate-path guard at generation time itself, not just in
    # validate.py -- catches the most common real mistake (running
    # metadata generation twice after a partial manual re-copy) immediately.
    dupes = df.duplicated(subset=["dataset", "video_path"], keep=False)
    if dupes.any():
        print(
            f"WARNING: {dupes.sum()} duplicate (dataset, video_path) rows "
            f"found — keeping only the first occurrence of each. Run "
            f"`python -m src.data.validate` after this for full details."
        )
        df = df.drop_duplicates(subset=["dataset", "video_path"], keep="first")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nMetadata written to {output_path}")
    print(f"Total videos: {len(df)}")
    print("Per-dataset counts:")
    print(df["dataset"].value_counts().to_string())
    print("Per-label counts:")
    print(df["label"].value_counts().to_string())

    return df


if __name__ == "__main__":
    generate_metadata()
