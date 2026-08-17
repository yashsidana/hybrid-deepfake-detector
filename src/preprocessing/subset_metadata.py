"""
Caps each enabled dataset's metadata rows at a target REAL/FAKE count, for
prototype runs that shouldn't ingest every video a dataset adapter finds
(e.g. an entire downloaded DFDC part, or all of Celeb-DF v2).

Run this AFTER `python -m src.data.metadata` (which scans every enabled
dataset's raw videos into data/metadata/metadata.csv) and BEFORE
`python -m src.preprocessing.create_splits` (which reads that same path).
No other script needs to change -- this just narrows the same file in
place, backing up the untouched original first.

Sampling is per (dataset, label) group, uniform random with a fixed seed
for reproducibility. This is a SELECTION step, not a split -- identity
grouping is still handled entirely by create_splits.py afterward, on
whatever rows remain here. A fake video's "original"/identity reference
pointing to a real video that got excluded by this subsetting is not a
leakage risk: create_splits.py's union-find only merges rows that are
BOTH present in the metadata it reads, so an excluded identity simply
means that token never appears, not that a phantom group forms.

If a (dataset, label) group has fewer videos available than requested,
ALL of them are kept and the shortfall is reported -- never silently
padded, duplicated, or invented. This matches the project's explicit
policy: exact target counts are a goal, not a requirement to force.
"""

import argparse
import os
import shutil

import pandas as pd

METADATA_PATH = "data/metadata/metadata.csv"
BACKUP_PATH = "data/metadata/metadata_full.csv"

LABEL_NAMES = {0: "real", 1: "fake"}


def parse_targets(specs):
    """
    specs: list of strings like "celebdf:1000:1000" (dataset:real:fake).
    Returns {dataset_name: {0: real_count, 1: fake_count}}.
    """
    targets = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Bad --target spec {spec!r}; expected dataset:real_count:fake_count, "
                f"e.g. celebdf:1000:1000"
            )
        name, real_n, fake_n = parts
        targets[name] = {0: int(real_n), 1: int(fake_n)}
    return targets


def subset(metadata_path, targets, seed=42, backup_path=BACKUP_PATH):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found. Run `python -m src.data.metadata` first."
        )

    df = pd.read_csv(metadata_path)

    if not os.path.exists(backup_path):
        shutil.copy(metadata_path, backup_path)
        print(f"Backed up full metadata ({len(df)} rows) to {backup_path}")
    else:
        print(
            f"{backup_path} already exists -- not overwriting. If you want "
            f"to re-subset from the true full set, restore it to "
            f"{metadata_path} first."
        )

    kept_frames = []
    report_rows = []

    for dataset_name, dataset_df in df.groupby("dataset"):
        if dataset_name not in targets:
            print(f"[{dataset_name}] no target given -- keeping ALL {len(dataset_df)} rows.")
            kept_frames.append(dataset_df)
            continue

        for label, target_n in targets[dataset_name].items():
            label_df = dataset_df[dataset_df["label"] == label]
            available = len(label_df)
            take_n = min(target_n, available)

            sampled = label_df.sample(n=take_n, random_state=seed) if take_n > 0 else label_df.iloc[0:0]
            kept_frames.append(sampled)

            report_rows.append({
                "dataset": dataset_name,
                "label": LABEL_NAMES.get(label, label),
                "requested": target_n,
                "available": available,
                "kept": take_n,
                "shortfall": max(0, target_n - available),
            })

    result = pd.concat(kept_frames, ignore_index=True) if kept_frames else df.iloc[0:0]
    result.to_csv(metadata_path, index=False)

    report_df = pd.DataFrame(report_rows)
    print("\nSubsetting report:")
    if len(report_df):
        print(report_df.to_string(index=False))
        shortfalls = report_df[report_df["shortfall"] > 0]
        if len(shortfalls):
            print(
                "\nNOTE: the following groups had fewer videos available than "
                "requested -- kept everything available rather than forcing "
                "the target:"
            )
            print(shortfalls.to_string(index=False))
    else:
        print("(no per-dataset targets matched any rows)")

    print(f"\n{metadata_path} now has {len(result)} rows (was {len(df)}).")
    print("Per-dataset/label counts after subsetting:")
    print(result.groupby(["dataset", "label"]).size().to_string())

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", action="append", required=True,
        help="dataset:real_count:fake_count, repeatable, e.g. "
             "--target celebdf:1000:1000 --target dfdc:1000:1000",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-path", default=METADATA_PATH)
    args = parser.parse_args()

    targets = parse_targets(args.target)
    subset(args.metadata_path, targets, seed=args.seed)


if __name__ == "__main__":
    main()
