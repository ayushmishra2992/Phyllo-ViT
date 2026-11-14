from pathlib import Path
from PIL import Image
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw"

CLASSES = {
    "Alternate": DATASET_DIR / "alternate",
    "Opposite": DATASET_DIR / "opposite",
    "Whorled": DATASET_DIR / "whorled",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


def get_hash(image_path):
    """Create a hash from the actual image file contents."""
    hasher = hashlib.md5()

    with open(image_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


hashes = {}
duplicates = []

for class_name, class_dir in CLASSES.items():

    for image_path in class_dir.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image_hash = get_hash(image_path)

        if image_hash in hashes:
            duplicates.append(
                (image_path, hashes[image_hash])
            )
        else:
            hashes[image_hash] = image_path


print("=" * 60)
print("DUPLICATE IMAGE AUDIT")
print("=" * 60)

print(f"\nUnique image files checked: {len(hashes)}")
print(f"Duplicate files found:     {len(duplicates)}")

if duplicates:
    print("\nDuplicates:")
    for duplicate, original in duplicates:
        print("\nDuplicate :", duplicate)
        print("Original  :", original)
else:
    print("\nNo exact duplicate images found.")

print("\n" + "=" * 60)