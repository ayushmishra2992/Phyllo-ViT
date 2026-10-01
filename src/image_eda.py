from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split"
    / "split_metadata.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "eda"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(METADATA_FILE)

print("=" * 60)
print("PHYLLO-ViT IMAGE EDA")
print("=" * 60)

print("\nDataset size:")
print(len(df))

# --------------------------------------------------
# 1. CLASS DISTRIBUTION
# --------------------------------------------------

class_counts = df["phyllotaxy"].value_counts()

print("\nClass distribution:")
print(class_counts)

plt.figure(figsize=(8, 5))

class_counts.plot(kind="bar")

plt.title("Phyllotaxy Class Distribution")
plt.xlabel("Phyllotaxy Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_distribution.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 2. IMAGE DIMENSIONS
# --------------------------------------------------

print("\nImage dimensions:")

print(
    df.groupby(["width", "height"])
    .size()
    .sort_values(ascending=False)
    .head(20)
)

plt.figure(figsize=(8, 5))

plt.scatter(
    df["width"],
    df["height"],
    alpha=0.7
)

plt.title("Image Resolution Distribution")
plt.xlabel("Width (pixels)")
plt.ylabel("Height (pixels)")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "resolution_distribution.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 3. IMAGES PER SPLIT
# --------------------------------------------------

split_counts = df["split"].value_counts()

print("\nSplit distribution:")
print(split_counts)

plt.figure(figsize=(8, 5))

split_counts.plot(kind="bar")

plt.title("Train / Validation / Test Distribution")
plt.xlabel("Dataset Split")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "split_distribution.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 4. CLASS DISTRIBUTION PER SPLIT
# --------------------------------------------------

split_class = pd.crosstab(
    df["split"],
    df["phyllotaxy"]
)

print("\nClass distribution per split:")
print(split_class)

ax = split_class.plot(
    kind="bar",
    figsize=(9, 5)
)

ax.set_title("Phyllotaxy Distribution Across Dataset Splits")
ax.set_xlabel("Dataset Split")
ax.set_ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_distribution_per_split.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 5. SAMPLE IMAGES
# --------------------------------------------------

sample_classes = [
    "Alternate",
    "Opposite",
    "Whorled"
]

fig, axes = plt.subplots(
    3,
    3,
    figsize=(10, 10)
)

for row, class_name in enumerate(sample_classes):

    class_images = df[
        (df["phyllotaxy"] == class_name) &
        (df["split"] == "train")
    ].head(3)

    for col, (_, record) in enumerate(
        class_images.iterrows()
    ):

        image_path = (
            PROJECT_ROOT
            / record["image_path"]
        )

        try:
            image = Image.open(image_path)

            axes[row, col].imshow(image)
            axes[row, col].axis("off")
            axes[row, col].set_title(class_name)

        except Exception as error:

            axes[row, col].text(
                0.5,
                0.5,
                "Image error",
                ha="center"
            )

            axes[row, col].axis("off")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "sample_images.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# DONE
# --------------------------------------------------

print("\nEDA completed successfully.")

print("\nGraphs saved in:")
print(RESULTS_DIR)

print("\nGenerated files:")

for file in RESULTS_DIR.iterdir():
    print(" -", file.name)

print("=" * 60)