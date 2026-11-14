from pathlib import Path
import shutil
import csv


# =========================================================
# Paths
# =========================================================

SOURCE_DIR = Path("data/raw/not_a_leaf")

DEST_DIR = Path(
    "data/processed/clean/Not_a_leaf"
)

METADATA_FILE = Path(
    "data/metadata/not_a_leaf_metadata.csv"
)


# =========================================================
# Main
# =========================================================

def main():

    print("=" * 60)
    print("Preparing Not_a_leaf Dataset")
    print("=" * 60)
    print()

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source directory not found: {SOURCE_DIR}"
        )

    files = sorted(
        SOURCE_DIR.glob("*.jpg")
    )

    print(
        f"Source images found: {len(files)}"
    )

    if not files:
        raise RuntimeError(
            "No JPG images found."
        )

    # Create destination
    DEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    copied = 0

    # -----------------------------------------------------
    # Copy images
    # -----------------------------------------------------

    for source_file in files:

        destination_file = (
            DEST_DIR / source_file.name
        )

        shutil.copy2(
            source_file,
            destination_file,
        )

        copied += 1

    print(
        f"Images copied: {copied}"
    )

    print(
        f"Destination: {DEST_DIR.resolve()}"
    )

    print()

    # -----------------------------------------------------
    # Create metadata
    # -----------------------------------------------------

    METADATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METADATA_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "label",
                "source",
                "source_category",
                "hard_negative",
                "split",
            ],
        )

        writer.writeheader()

        for image_file in files:

            writer.writerow(
                {
                    "filename": image_file.name,
                    "label": "Not_a_leaf",
                    "source": "COCO2017",
                    "source_category": "",
                    "hard_negative": "False",
                    "split": "",
                }
            )

    print(
        f"Metadata saved: {METADATA_FILE}"
    )

    print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print()

    print(
        "IMPORTANT:"
    )

    print(
        "The original images remain untouched in:"
    )

    print(
        f"  {SOURCE_DIR}"
    )

    print()

    print(
        "Prepared images are in:"
    )

    print(
        f"  {DEST_DIR}"
    )

    print()

    print(
        "Next we will create the 4-class train/validation/test split."
    )


if __name__ == "__main__":
    main()