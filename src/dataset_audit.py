from pathlib import Path
from PIL import Image

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw"

CLASSES = {
    "Alternate": DATASET_DIR / "alternate",
    "Opposite": DATASET_DIR / "opposite",
    "Whorled": DATASET_DIR / "whorled",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

print("=" * 60)
print("PHYLLO-ViT DATASET AUDIT")
print("=" * 60)

total_images = 0

for class_name, class_dir in CLASSES.items():

    print(f"\n{class_name}")
    print("-" * 40)

    if not class_dir.exists():
        print("ERROR: Folder not found:", class_dir)
        continue

    images = [
        file for file in class_dir.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    valid_images = 0
    corrupted_images = 0
    resolutions = set()

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.verify()

            # Reopen after verify
            with Image.open(image_path) as img:
                resolutions.add(img.size)

            valid_images += 1

        except Exception:
            corrupted_images += 1
            print("Corrupted:", image_path)

    print("Total image files :", len(images))
    print("Valid images      :", valid_images)
    print("Corrupted images  :", corrupted_images)
    print("Unique resolutions:", len(resolutions))

    total_images += valid_images

print("\n" + "=" * 60)
print("TOTAL VALID IMAGES:", total_images)
print("=" * 60)