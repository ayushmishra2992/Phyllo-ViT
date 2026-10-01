from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "metadata.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split"
)

# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

df = pd.read_csv(METADATA_FILE)

# Keep only unique images
df = df[df["is_duplicate"] == False].copy()

# Reset index
df = df.reset_index(drop=True)

print("=" * 60)
print("PHYLLO-ViT DATASET SPLIT")
print("=" * 60)

print(f"\nTotal unique images: {len(df)}")

print("\nOriginal class distribution:")
print(df["phyllotaxy"].value_counts())

# --------------------------------------------------
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["phyllotaxy"],
    random_state=42
)

# --------------------------------------------------
# SECOND SPLIT
# 15% VALIDATION
# 15% TEST
# --------------------------------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["phyllotaxy"],
    random_state=42
)

# Add split labels
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

# Combine
split_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

# --------------------------------------------------
# SAVE SPLIT METADATA
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

split_csv = OUTPUT_DIR / "split_metadata.csv"

split_df.to_csv(
    split_csv,
    index=False
)

# --------------------------------------------------
# CREATE FOLDERS + COPY IMAGES
# --------------------------------------------------

for split_name, split_data in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df)
]:

    for class_name in [
        "Alternate",
        "Opposite",
        "Whorled"
    ]:

        class_data = split_data[
            split_data["phyllotaxy"] == class_name
        ]

        destination_dir = (
            OUTPUT_DIR
            / split_name
            / class_name
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for _, row in class_data.iterrows():

            source = PROJECT_ROOT / row["image_path"]

            destination = (
                destination_dir
                / row["filename"]
            )

            shutil.copy2(
                source,
                destination
            )

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\nSplit sizes:")
print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

print("\nTrain distribution:")
print(train_df["phyllotaxy"].value_counts())

print("\nValidation distribution:")
print(val_df["phyllotaxy"].value_counts())

print("\nTest distribution:")
print(test_df["phyllotaxy"].value_counts())

print("\nSaved split metadata:")
print(split_csv)

print("\nDataset split completed successfully.")

print("=" * 60)