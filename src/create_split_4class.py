from pathlib import Path
import shutil
import random
import csv


# =========================================================
# Configuration
# =========================================================

SOURCE_DIR = Path("data/processed/clean")

OUTPUT_DIR = Path(
    "data/processed/split_4class"
)

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASSES = [
    "Alternate",
    "Opposite",
    "Whorled",
    "Not_a_leaf",
]


# =========================================================
# Main
# =========================================================

def main():

    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("Phyllo-ViT — 4-Class Dataset Split")
    print("=" * 60)
    print()

    print("Classes:")

    for class_name in CLASSES:
        print(f"  - {class_name}")

    print()

    # -----------------------------------------------------
    # Check source directories
    # -----------------------------------------------------

    for class_name in CLASSES:

        class_dir = SOURCE_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

    # -----------------------------------------------------
    # Remove previous 4-class split
    # -----------------------------------------------------

    if OUTPUT_DIR.exists():

        print(
            "Existing 4-class split found."
        )

        print(
            "Removing it before recreating..."
        )

        shutil.rmtree(OUTPUT_DIR)

    # -----------------------------------------------------
    # Create output directories
    # -----------------------------------------------------

    for split in [
        "train",
        "val",
        "test",
    ]:

        for class_name in CLASSES:

            (
                OUTPUT_DIR
                / split
                / class_name
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata_rows = []

    # -----------------------------------------------------
    # Process each class
    # -----------------------------------------------------

    for class_name in CLASSES:

        source_class_dir = (
            SOURCE_DIR / class_name
        )

        image_files = sorted(
            [
                f
                for f in source_class_dir.iterdir()
                if f.is_file()
                and f.suffix.lower()
                in [".jpg", ".jpeg", ".png"]
            ]
        )

        if not image_files:
            raise RuntimeError(
                f"No images found for {class_name}"
            )

        random.shuffle(image_files)

        total = len(image_files)

        train_count = int(
            total * TRAIN_RATIO
        )

        val_count = int(
            total * VAL_RATIO
        )

        test_count = (
            total
            - train_count
            - val_count
        )

        train_files = image_files[
            :train_count
        ]

        val_files = image_files[
            train_count:
            train_count + val_count
        ]

        test_files = image_files[
            train_count + val_count:
        ]

        split_files = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        print(
            f"{class_name}:"
        )

        print(
            f"  Total: {total}"
        )

        print(
            f"  Train: {len(train_files)}"
        )

        print(
            f"  Val:   {len(val_files)}"
        )

        print(
            f"  Test:  {len(test_files)}"
        )

        print()

        # -------------------------------------------------
        # Copy files
        # -------------------------------------------------

        for split, files in split_files.items():

            for source_file in files:

                destination_file = (
                    OUTPUT_DIR
                    / split
                    / class_name
                    / source_file.name
                )

                shutil.copy2(
                    source_file,
                    destination_file,
                )

                metadata_rows.append(
                    {
                        "filename": source_file.name,
                        "class": class_name,
                        "split": split,
                        "source_path": str(
                            source_file
                        ),
                    }
                )

    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------

    metadata_path = (
        OUTPUT_DIR
        / "split_metadata.csv"
    )

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "class",
                "split",
                "source_path",
            ],
        )

        writer.writeheader()

        writer.writerows(
            metadata_rows
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("=" * 60)
    print("4-Class Split Complete")
    print("=" * 60)
    print()

    print(
        f"Total images: {len(metadata_rows)}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print()

    print(
        "Metadata:"
    )

    print(
        metadata_path.resolve()
    )

    print()

    # -----------------------------------------------------
    # Print final split totals
    # -----------------------------------------------------

    for split in [
        "train",
        "val",
        "test",
    ]:

        count = sum(
            1
            for row in metadata_rows
            if row["split"] == split
        )

        print(
            f"{split}: {count}"
        )

    print()

    print(
        "The original 3-class split was NOT modified."
    )


if __name__ == "__main__":
    main()