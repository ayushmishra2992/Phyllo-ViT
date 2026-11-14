import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor, ViTForImageClassification


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "split")
MODEL_DIR = os.path.join(BASE_DIR, "models", "vit_phyllotaxy")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "vit_embeddings")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 8


print("=" * 60)
print("Phyllo-ViT — Embedding Extraction")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD PROCESSOR + TRAINED MODEL
# ============================================================

print("\nLoading trained ViT...")

processor = ViTImageProcessor.from_pretrained(
    MODEL_DIR
)

model = ViTForImageClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()

print("Trained model loaded successfully.")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

def vit_transform(image):

    processed = processor(
        images=image,
        return_tensors="pt"
    )

    return processed["pixel_values"].squeeze(0)


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

def extract_split(split_name):

    split_dir = os.path.join(
        DATA_DIR,
        split_name
    )

    dataset = ImageFolder(
        split_dir,
        transform=vit_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    embeddings = []
    labels = []
    paths = []

    print(
        f"\nExtracting {split_name} embeddings "
        f"from {len(dataset)} images..."
    )

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            # Access the ViT backbone directly.
            # This gives us the learned visual representation.
            outputs = model.vit(
                pixel_values=images
            )

            # CLS token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            embeddings.append(
                cls_embeddings.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    labels = np.array(labels)

    # ImageFolder stores the class names in class_to_idx
    idx_to_class = {
        value: key
        for key, value in dataset.class_to_idx.items()
    }

    class_names = [
        idx_to_class[label]
        for label in labels
    ]

    # Recover image paths in exactly the same order
    image_paths = [
        path
        for path, _ in dataset.samples
    ]

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    embedding_file = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_embeddings.npy"
    )

    np.save(
        embedding_file,
        embeddings
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = pd.DataFrame({
        "image_path": image_paths,
        "phyllotaxy": class_names,
        "label": labels
    })

    metadata_file = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_embedding_metadata.csv"
    )

    metadata.to_csv(
        metadata_file,
        index=False
    )

    print(
        f"{split_name}: "
        f"{embeddings.shape[0]} images × "
        f"{embeddings.shape[1]} embedding dimensions"
    )

    print(f"Saved: {embedding_file}")
    print(f"Saved: {metadata_file}")

    return embeddings


# ============================================================
# EXTRACT ALL SPLITS
# ============================================================

train_embeddings = extract_split("train")
val_embeddings = extract_split("val")
test_embeddings = extract_split("test")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDING EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Train embeddings: {train_embeddings.shape}"
)

print(
    f"Validation embeddings: {val_embeddings.shape}"
)

print(
    f"Test embeddings: {test_embeddings.shape}"
)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)