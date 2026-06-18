import os
import shutil
import kagglehub


def setup_celebdf():
    path = kagglehub.dataset_download("reubensuju/celeb-df-v2")

    target_root = "data/raw/celebdf"
    real_target = os.path.join(target_root, "real")
    fake_target = os.path.join(target_root, "fake")

    os.makedirs(real_target, exist_ok=True)
    os.makedirs(fake_target, exist_ok=True)

    # inspect downloaded folders
    for root, dirs, files in os.walk(path):
        for file in files:
            src = os.path.join(root, file)

            if file.endswith(".mp4"):
                root_lower = root.lower()

                if "real" in root_lower:
                    shutil.copy(src, real_target)

                elif "fake" in root_lower or "synthesis" in root_lower:
                    shutil.copy(src, fake_target)

    print("Celeb-DF dataset setup complete")


if __name__ == "__main__":
    setup_celebdf()