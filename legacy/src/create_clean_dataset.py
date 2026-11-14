from pathlib import Path
import pandas as pd
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "clean"

df = pd.read_csv(METADATA_FILE)

# Keep only unique images
clean_df = df[df["is_duplicate"] == False].copy()

print("=" * 60)
print("CREATING CLEAN DATASET")
print("=" * 60)

print(f"\nOriginal images : {len(df)}")
print(f"Unique images   : {len(clean_df)}")
print(f"Removed         : {len(df) - len(clean_df)}")

for _, row in clean_df.iterrows():

    source = PROJECT_ROOT / row["image_path"]

    class_name = row["phyllotaxy"]

    destination_dir = OUTPUT_DIR / class_name
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / row["filename"]

    shutil.copy2(source, destination)

print("\nClean dataset created.")

print("\nClass distribution:")
print(clean_df["phyllotaxy"].value_counts())

print("\nLocation:")
print(OUTPUT_DIR)

print("=" * 60)