"""
Validates data/metadata/metadata.csv before any preprocessing runs.
Produces a human-readable report and a machine-readable JSON summary at
data/metadata/validation_report.json — run this after metadata generation
and before precompute_faces.py / precompute_temporal.py.

Checks performed:
  - missing files          (metadata references a path that doesn't exist)
  - corrupted videos       (file exists but OpenCV can't read fps/frames)
  - duplicate filenames    (same filename appears more than once, possibly
                             across different dataset/folder -- not
                             necessarily a bug, but worth knowing about)
  - duplicate video rows   (identical (dataset, video_path) -- an actual
                             bug if found here, since metadata.py already
                             de-duplicates at generation time)
  - incorrect labels       (label not in {0, 1})
  - missing metadata       (null/blank required fields)
  - invalid frame counts   (number_of_frames missing, zero, or negative)
"""

import json
import os

import pandas as pd
import yaml

METADATA_PATH = "data/metadata/metadata.csv"
REPORT_PATH = "data/metadata/validation_report.json"
CONFIG_PATH = "config/config.yaml"


def _resolve_abs_path(row, raw_root):
    return os.path.join(raw_root, row["dataset"], row["video_path"])


def validate(metadata_path=METADATA_PATH, raw_root=None, report_path=REPORT_PATH):
    if raw_root is None:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)
        raw_root = config.get("data", {}).get("raw_dir", "data/raw")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"{metadata_path} not found. Run `python -m src.data.metadata` first."
        )

    df = pd.read_csv(metadata_path)
    issues = {}

    # --- missing files ---
    df["_abs_path"] = df.apply(lambda r: _resolve_abs_path(r, raw_root), axis=1)
    missing_mask = ~df["_abs_path"].apply(os.path.exists)
    issues["missing_files"] = df.loc[missing_mask, ["dataset", "video_path"]].to_dict("records")

    # --- incorrect labels ---
    bad_label_mask = ~df["label"].isin([0, 1])
    issues["incorrect_labels"] = df.loc[bad_label_mask, ["dataset", "video_path", "label"]].to_dict("records")

    # --- invalid frame counts ---
    invalid_frames_mask = df["number_of_frames"].isna() | (df["number_of_frames"].fillna(0) <= 0)
    issues["invalid_frame_counts"] = df.loc[invalid_frames_mask & ~missing_mask, ["dataset", "video_path", "number_of_frames"]].to_dict("records")

    # --- corrupted videos: file exists but core probe fields are all null ---
    corrupted_mask = (~missing_mask) & df["fps"].isna() & df["duration"].isna() & df["resolution"].isna()
    issues["corrupted_videos"] = df.loc[corrupted_mask, ["dataset", "video_path"]].to_dict("records")

    # --- missing metadata: required fields blank ---
    required_cols = ["dataset", "video_path", "label"]
    missing_meta_mask = df[required_cols].isna().any(axis=1)
    issues["missing_metadata"] = df.loc[missing_meta_mask, required_cols].to_dict("records")

    # --- duplicate (dataset, video_path) rows ---
    dup_rows_mask = df.duplicated(subset=["dataset", "video_path"], keep=False)
    issues["duplicate_video_rows"] = df.loc[dup_rows_mask, ["dataset", "video_path"]].to_dict("records")

    # --- duplicate filenames (informational, not necessarily a bug) ---
    df["_filename"] = df["video_path"].apply(os.path.basename)
    dup_fname_mask = df.duplicated(subset=["_filename"], keep=False)
    dup_filenames = (
        df.loc[dup_fname_mask, ["dataset", "video_path", "_filename"]]
        .to_dict("records")
    )
    issues["duplicate_filenames"] = dup_filenames

    total_issues = sum(len(v) for k, v in issues.items() if k != "duplicate_filenames")

    summary = {
        "total_videos_checked": len(df),
        "total_blocking_issues": total_issues,
        "issue_counts": {k: len(v) for k, v in issues.items()},
        "details": issues,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("=" * 55)
    print("DATASET VALIDATION REPORT")
    print("=" * 55)
    print(f"Total videos checked: {len(df)}")
    for check_name, records in issues.items():
        flag = "  " if check_name == "duplicate_filenames" else "⚠ "
        print(f"{flag}{check_name}: {len(records)}")
    print(f"\nFull report saved to {report_path}")

    if total_issues > 0:
        print(
            "\nBlocking issues found (everything above except "
            "duplicate_filenames, which is informational only). Review "
            f"{report_path} before running precompute_faces.py / "
            "precompute_temporal.py — videos with missing files or "
            "invalid labels will otherwise fail or silently skew training."
        )
    else:
        print("\nNo blocking issues found. Safe to proceed to preprocessing.")

    return summary


if __name__ == "__main__":
    validate()
