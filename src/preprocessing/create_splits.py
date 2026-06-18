import os
import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(dataset_root):
    records = []

    # Process real videos
    real_dir = os.path.join(dataset_root, "real")
    for video in os.listdir(real_dir):
        if video.endswith(".mp4"):
            records.append({
                "video": os.path.join("real", video),
                "label": 0
            })

    # Process fake videos
    fake_dir = os.path.join(dataset_root, "fake")
    for video in os.listdir(fake_dir):
        if video.endswith(".mp4"):
            records.append({
                "video": os.path.join("fake", video),
                "label": 1
            })

    df = pd.DataFrame(records)

    # Train: 80%, Temp: 20%
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    # Validation: 10%, Test: 10%
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42
    )

    os.makedirs("data/splits", exist_ok=True)

    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv", index=False)
    test_df.to_csv("data/splits/test.csv", index=False)

    print("Splits created successfully.")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    create_splits("data/raw/celebdf")