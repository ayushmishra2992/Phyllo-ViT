from pathlib import Path
import json
import random
import zipfile
import warnings

import requests
from PIL import Image
from io import BytesIO


# =========================================================
# Configuration
# =========================================================

OUTPUT_DIR = Path("data/raw/not_a_leaf")
TEMP_DIR = Path("data/raw/coco_temp")

TARGET_IMAGES = 150
RANDOM_SEED = 42

ANNOTATIONS_URL = (
    "https://images.cocodataset.org/annotations/"
    "annotations_trainval2017.zip"
)

ANNOTATIONS_ZIP = Path("data/raw/coco_annotations.zip")

# COCO categories that are useful as "non-leaf" examples.
#
# We intentionally focus mainly on people, vehicles,
# animals, indoor objects and everyday objects.
#
# Plant-related categories are NOT included.
CATEGORIES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "bottle",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# =========================================================
# SSL configuration
# =========================================================
#
# Your system previously produced:
#
# SSL: CERTIFICATE_VERIFY_FAILED
# hostname mismatch: images.cocodataset.org
#
# Therefore this script allows the public COCO download
# through requests without certificate verification.
#
# This is only being used for downloading public dataset
# files. We do NOT send any personal information.
#
# =========================================================

warnings.filterwarnings("ignore")

VERIFY_SSL = False


# =========================================================
# Helper: HTTP GET
# =========================================================

def get_response(url: str, stream: bool = False, timeout: int = 60):
    """
    Download a URL while handling the SSL certificate problem
    encountered on this Windows environment.
    """

    response = requests.get(
        url,
        stream=stream,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
        verify=VERIFY_SSL,
    )

    response.raise_for_status()

    return response


# =========================================================
# Download COCO annotations
# =========================================================

def download_annotations():
    """
    Download the COCO 2017 train/validation annotations.

    This file is much smaller than the full COCO image archive.
    """

    ANNOTATIONS_ZIP.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if ANNOTATIONS_ZIP.exists():
        print("COCO annotations already exist.")
        return

    print()
    print("=" * 60)
    print("Downloading COCO annotations")
    print("=" * 60)
    print()

    print("URL:")
    print(ANNOTATIONS_URL)
    print()

    response = get_response(
        ANNOTATIONS_URL,
        stream=True,
        timeout=120,
    )

    total_size = int(
        response.headers.get("content-length", 0)
    )

    downloaded = 0

    with open(ANNOTATIONS_ZIP, "wb") as f:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if not chunk:
                continue

            f.write(chunk)

            downloaded += len(chunk)

            if total_size:
                percent = (
                    downloaded / total_size
                ) * 100

                print(
                    f"\rDownloaded: {percent:6.2f}%",
                    end="",
                )

    print()
    print()
    print("COCO annotations downloaded successfully.")
    print()


# =========================================================
# Extract annotation JSON
# =========================================================

def extract_annotations():
    """
    Extract only the COCO training annotation JSON.
    """

    TEMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_json = (
        TEMP_DIR
        / "instances_train2017.json"
    )

    if annotation_json.exists():
        print("Annotation JSON already extracted.")
        return annotation_json

    print("=" * 60)
    print("Extracting COCO annotations")
    print("=" * 60)
    print()

    with zipfile.ZipFile(
        ANNOTATIONS_ZIP,
        "r",
    ) as zip_file:

        target = (
            "annotations/"
            "instances_train2017.json"
        )

        if target not in zip_file.namelist():
            raise FileNotFoundError(
                "instances_train2017.json "
                "was not found inside the COCO archive."
            )

        with zip_file.open(target) as source:
            with open(
                annotation_json,
                "wb",
            ) as destination:

                destination.write(
                    source.read()
                )

    print("Annotation JSON extracted.")
    print()

    return annotation_json


# =========================================================
# Read COCO annotations
# =========================================================

def load_annotations(annotation_json):
    """
    Load COCO annotations.
    """

    print("=" * 60)
    print("Reading COCO annotations")
    print("=" * 60)
    print()

    with open(
        annotation_json,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    print(
        f"Images available: {len(data['images'])}"
    )

    print(
        f"Annotations available: "
        f"{len(data['annotations'])}"
    )

    print(
        f"Categories available: "
        f"{len(data['categories'])}"
    )

    print()

    return data


# =========================================================
# Select image IDs
# =========================================================

def select_images(data):
    """
    Select unique COCO images belonging to the
    requested non-leaf categories.
    """

    category_id_to_name = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    selected_category_ids = {
        category_id
        for category_id, category_name
        in category_id_to_name.items()
        if category_name in CATEGORIES
    }

    print(
        "Selected COCO categories:"
    )

    for category_id in sorted(
        selected_category_ids
    ):
        print(
            f"  - {category_id_to_name[category_id]}"
        )

    print()

    # Map image ID -> set of categories
    image_categories = {}

    for annotation in data["annotations"]:

        category_id = annotation["category_id"]

        if category_id not in selected_category_ids:
            continue

        image_id = annotation["image_id"]

        if image_id not in image_categories:
            image_categories[image_id] = set()

        image_categories[image_id].add(
            category_id_to_name[category_id]
        )

    # Image metadata lookup
    image_lookup = {
        image["id"]: image
        for image in data["images"]
    }

    candidates = []

    for image_id, categories in image_categories.items():

        if image_id not in image_lookup:
            continue

        image_info = image_lookup[image_id]

        candidates.append(
            {
                "id": image_id,
                "file_name": image_info["file_name"],
                "width": image_info["width"],
                "height": image_info["height"],
                "categories": sorted(categories),
            }
        )

    print(
        f"Candidate non-leaf images found: "
        f"{len(candidates)}"
    )

    print()

    # Reproducible selection
    random.seed(RANDOM_SEED)
    random.shuffle(candidates)

    return candidates[:TARGET_IMAGES]


# =========================================================
# Download one image
# =========================================================

def download_image(
    image_info,
    output_path,
):
    """
    Download and validate a single COCO image.
    """

    image_url = (
        "https://images.cocodataset.org/train2017/"
        + image_info["file_name"]
    )

    try:

        response = get_response(
            image_url,
            stream=False,
            timeout=30,
        )

        image = Image.open(
            BytesIO(response.content)
        )

        image = image.convert("RGB")

        # Avoid extremely tiny images
        if image.width < 100:
            return False

        if image.height < 100:
            return False

        image.save(
            output_path,
            "JPEG",
            quality=95,
        )

        return True

    except Exception as e:

        print(
            f"\nFailed: "
            f"{image_info['file_name']} "
            f"-> {e}"
        )

        return False


# =========================================================
# Download selected images
# =========================================================

def download_selected_images(
    selected_images
):
    """
    Download the selected COCO images.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("Downloading selected images")
    print("=" * 60)
    print()

    successful = 0
    failed = 0

    for index, image_info in enumerate(
        selected_images,
        start=1,
    ):

        # Sequential filename makes our dataset
        # independent of COCO's original filename.
        output_filename = (
            f"not_a_leaf_{index:04d}.jpg"
        )

        output_path = (
            OUTPUT_DIR
            / output_filename
        )

        # Skip if already downloaded
        if output_path.exists():

            successful += 1

            print(
                f"\r[{index}/{len(selected_images)}] "
                f"Already exists",
                end="",
            )

            continue

        success = download_image(
            image_info,
            output_path,
        )

        if success:

            successful += 1

            print(
                f"\r[{index}/{len(selected_images)}] "
                f"Downloaded: "
                f"{output_filename}",
                end="",
            )

        else:

            failed += 1

            print(
                f"\r[{index}/{len(selected_images)}] "
                f"Failed",
                end="",
            )

    print()
    print()

    print("=" * 60)
    print("Download summary")
    print("=" * 60)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Output:     {OUTPUT_DIR.resolve()}"
    )

    print()


# =========================================================
# Save source metadata
# =========================================================

def save_metadata(selected_images):
    """
    Save metadata describing where each negative image
    came from.

    This is important for reproducibility and for the
    project report.
    """

    metadata_path = (
        Path("data/metadata")
        / "not_a_leaf_metadata.csv"
    )

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for index, image_info in enumerate(
        selected_images,
        start=1,
    ):

        output_filename = (
            f"not_a_leaf_{index:04d}.jpg"
        )

        output_path = (
            OUTPUT_DIR
            / output_filename
        )

        if not output_path.exists():
            continue

        source_categories = "|".join(
            image_info["categories"]
        )

        rows.append(
            {
                "filename": output_filename,
                "label": "Not_a_leaf",
                "source": "COCO2017",
                "source_category": source_categories,
                "hard_negative": False,
                "split": "",
            }
        )

    import csv

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
                "label",
                "source",
                "source_category",
                "hard_negative",
                "split",
            ],
        )

        writer.writeheader()

        writer.writerows(rows)

    print(
        f"Metadata saved to: "
        f"{metadata_path}"
    )

    print(
        f"Metadata records: {len(rows)}"
    )

    print()


# =========================================================
# Main
# =========================================================

def main():

    print()
    print("=" * 60)
    print("Phyllo-ViT")
    print("Not-a-Leaf Dataset Builder")
    print("=" * 60)
    print()

    print(
        f"Target images: {TARGET_IMAGES}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print()

    # -----------------------------------------------------
    # Step 1
    # Download annotations
    # -----------------------------------------------------

    download_annotations()

    # -----------------------------------------------------
    # Step 2
    # Extract annotations
    # -----------------------------------------------------

    annotation_json = (
        extract_annotations()
    )

    # -----------------------------------------------------
    # Step 3
    # Load annotations
    # -----------------------------------------------------

    data = load_annotations(
        annotation_json
    )

    # -----------------------------------------------------
    # Step 4
    # Select candidate images
    # -----------------------------------------------------

    selected_images = select_images(
        data
    )

    if not selected_images:
        raise RuntimeError(
            "No suitable COCO images were found."
        )

    # -----------------------------------------------------
    # Step 5
    # Download selected images
    # -----------------------------------------------------

    download_selected_images(
        selected_images
    )

    # -----------------------------------------------------
    # Step 6
    # Save metadata
    # -----------------------------------------------------

    save_metadata(
        selected_images
    )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print()

    print(
        "Not-a-leaf images are now in:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    print()

    print(
        "Next step: inspect the downloaded images "
        "before adding them to the training split."
    )

    print()


if __name__ == "__main__":
    main()