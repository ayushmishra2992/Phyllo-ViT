from pathlib import Path
from PIL import Image
import pandas as pd
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = {
    "Alternate": DATASET_DIR / "alternate",
    "Opposite": DATASET_DIR / "opposite",
    "Whorled": DATASET_DIR / "whorled",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}


def calculate_hash(image_path):
    """Calculate MD5 hash of the image file."""
    hasher = hashlib.md5()

    with open(image_path, "rb") as file:
        while chunk := file.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


records = []
hash_to_first_image = {}

for class_name, class_dir in CLASSES.items():

    for image_path in sorted(class_dir.rglob("*")):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            # Read image information
            with Image.open(image_path) as image:
                width, height = image.size
                image_format = image.format

                # Verify image is readable
                image.verify()

            # Hash
            image_hash = calculate_hash(image_path)

            # Duplicate check
            if image_hash in hash_to_first_image:
                is_duplicate = True
                duplicate_of = hash_to_first_image[image_hash]
            else:
                is_duplicate = False
                duplicate_of = ""
                hash_to_first_image[image_hash] = str(image_path)

            records.append({
                "image_id": len(records) + 1,
                "filename": image_path.name,
                "image_path": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "phyllotaxy": class_name,
                "width": width,
                "height": height,
                "format": image_format,
                "file_size_bytes": image_path.stat().st_size,
                "image_hash": image_hash,
                "is_duplicate": is_duplicate,
                "duplicate_of": duplicate_of
            })

        except Exception as error:

            records.append({
                "image_id": len(records) + 1,
                "filename": image_path.name,
                "image_path": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
                "phyllotaxy": class_name,
                "width": None,
                "height": None,
                "format": None,
                "file_size_bytes": image_path.stat().st_size,
                "image_hash": None,
                "is_duplicate": False,
                "duplicate_of": "",
            })

df = pd.DataFrame(records)

output_file = OUTPUT_DIR / "metadata.csv"
df.to_csv(output_file, index=False)

print("=" * 60)
print("PHYLLO-ViT METADATA CREATED")
print("=" * 60)

print(f"\nTotal records: {len(df)}")

print("\nClass distribution:")
print(df["phyllotaxy"].value_counts())

print("\nDuplicate status:")
print(df["is_duplicate"].value_counts())

print("\nUnique images:")
print(df["image_hash"].nunique())

print(f"\nSaved to:")
print(output_file)

print("\n" + "=" * 60)