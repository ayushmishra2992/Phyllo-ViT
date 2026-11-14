from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
)
from tqdm import tqdm


# =========================================================
# Configuration
# =========================================================

DATA_DIR = Path(
    "data/processed/split_4class"
)

MODEL_DIR = Path(
    "models/vit_phyllotaxy_4class"
)

RESULTS_DIR = Path(
    "results/vit_4class"
)

PRETRAINED_MODEL = (
    "google/vit-base-patch16-224-in21k"
)

CLASSES = [
    "Alternate",
    "Opposite",
    "Whorled",
    "Not_a_leaf",
]

LABEL2ID = {
    label: index
    for index, label in enumerate(CLASSES)
}

ID2LABEL = {
    index: label
    for index, label in enumerate(CLASSES)
}

BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

RANDOM_SEED = 42


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Dataset
# =========================================================

class PhyllotaxyDataset(Dataset):

    def __init__(
        self,
        split,
        processor,
    ):

        self.split = split

        self.processor = processor

        self.samples = []

        split_dir = (
            DATA_DIR / split
        )

        for class_name in CLASSES:

            class_dir = (
                split_dir / class_name
            )

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: {class_dir}"
                )

            for image_path in sorted(
                class_dir.iterdir()
            ):

                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in [".jpg", ".jpeg", ".png"]
                ):

                    self.samples.append(
                        (
                            image_path,
                            LABEL2ID[class_name],
                        )
                    )

        if not self.samples:
            raise RuntimeError(
                f"No images found for split: {split}"
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = (
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        encoded = self.processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = (
            encoded["pixel_values"]
            .squeeze(0)
        )

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(
                label,
                dtype=torch.long,
            ),
        }


# =========================================================
# Class weights
# =========================================================

def calculate_class_weights(dataset):

    counts = np.zeros(
        len(CLASSES),
        dtype=np.float32,
    )

    for _, label in dataset.samples:
        counts[label] += 1

    # Inverse-frequency weighting.
    # This prevents the larger Not_a_leaf class
    # from dominating the loss.

    weights = (
        counts.sum()
        / (
            len(CLASSES)
            * counts
        )
    )

    return (
        torch.tensor(
            weights,
            dtype=torch.float32,
        ),
        counts,
    )


# =========================================================
# Evaluation
# =========================================================

def evaluate(
    model,
    dataloader,
    device,
    criterion,
):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for batch in dataloader:

            pixel_values = (
                batch["pixel_values"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            outputs = model(
                pixel_values=pixel_values,
            )

            loss = criterion(
                outputs.logits,
                labels,
            )

            predictions = (
                torch.argmax(
                    outputs.logits,
                    dim=1,
                )
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = (
        total_loss / total_samples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": all_labels,
        "predictions": all_predictions,
    }


# =========================================================
# Main
# =========================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("Phyllo-ViT — 4-Class ViT Training")
    print("=" * 70)
    print()

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print()

    print("Classes:")

    for index, class_name in enumerate(
        CLASSES
    ):

        print(
            f"  {index}: {class_name}"
        )

    print()

    # -----------------------------------------------------
    # Processor
    # -----------------------------------------------------

    print(
        "Loading pretrained ViT..."
    )

    processor = (
        ViTImageProcessor.from_pretrained(
            PRETRAINED_MODEL
        )
    )

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    print(
        "Loading datasets..."
    )

    train_dataset = PhyllotaxyDataset(
        "train",
        processor,
    )

    val_dataset = PhyllotaxyDataset(
        "val",
        processor,
    )

    test_dataset = PhyllotaxyDataset(
        "test",
        processor,
    )

    print()

    print(
        f"Train images: {len(train_dataset)}"
    )

    print(
        f"Validation images: {len(val_dataset)}"
    )

    print(
        f"Test images: {len(test_dataset)}"
    )

    print()

    # -----------------------------------------------------
    # Class distribution
    # -----------------------------------------------------

    class_weights, train_counts = (
        calculate_class_weights(
            train_dataset
        )
    )

    print(
        "Training class distribution:"
    )

    for index, class_name in enumerate(
        CLASSES
    ):

        print(
            f"  {class_name}: "
            f"{int(train_counts[index])}"
        )

    print()

    print(
        "Class weights:"
    )

    for index, class_name in enumerate(
        CLASSES
    ):

        print(
            f"  {class_name}: "
            f"{class_weights[index].item():.4f}"
        )

    print()

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    print(
        "Loading pretrained ViT classifier..."
    )

    model = (
        ViTForImageClassification.from_pretrained(
            PRETRAINED_MODEL,
            num_labels=len(CLASSES),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
    )

    model.to(device)

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    class_weights = class_weights.to(
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -----------------------------------------------------
    # Mixed precision
    # -----------------------------------------------------

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # -----------------------------------------------------
    # Output directories
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_val_accuracy = -1.0

    training_history = []

    print()
    print("=" * 70)
    print("Starting training")
    print("=" * 70)
    print()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0
        processed_samples = 0

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch}/{EPOCHS}"
            ),
        )

        for batch in progress:

            pixel_values = (
                batch["pixel_values"]
                .to(device)
            )

            labels = (
                batch["labels"]
                .to(device)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                outputs = model(
                    pixel_values=pixel_values,
                )

                loss = criterion(
                    outputs.logits,
                    labels,
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_size = (
                labels.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            processed_samples += (
                batch_size
            )

            progress.set_postfix(
                loss=(
                    running_loss
                    / processed_samples
                )
            )

        train_loss = (
            running_loss
            / processed_samples
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        print()

        print(
            f"Epoch {epoch}:"
        )

        print(
            f"  Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"  Val Loss: "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"  Val Accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"  Val F1: "
            f"{val_metrics['f1']:.4f}"
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics[
                "accuracy"
            ],
            "val_precision": val_metrics[
                "precision"
            ],
            "val_recall": val_metrics[
                "recall"
            ],
            "val_f1": val_metrics[
                "f1"
            ],
        }

        training_history.append(
            history_row
        )

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------

        if (
            val_metrics["accuracy"]
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_metrics["accuracy"]
            )

            print()
            print(
                "New best validation model."
            )

            model.save_pretrained(
                MODEL_DIR
            )

            processor.save_pretrained(
                MODEL_DIR
            )

            with open(
                MODEL_DIR
                / "class_mapping.json",
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    {
                        "classes": CLASSES,
                        "label2id": LABEL2ID,
                        "id2label": {
                            str(k): v
                            for k, v
                            in ID2LABEL.items()
                        },
                    },
                    f,
                    indent=2,
                )

        print()

    # -----------------------------------------------------
    # Save training history
    # -----------------------------------------------------

    with open(
        RESULTS_DIR
        / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            training_history,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Load best model
    # -----------------------------------------------------

    print("=" * 70)
    print("Loading best validation model")
    print("=" * 70)
    print()

    best_model = (
        ViTForImageClassification
        .from_pretrained(
            MODEL_DIR
        )
    )

    best_model.to(device)

    # -----------------------------------------------------
    # Final test
    # -----------------------------------------------------

    test_metrics = evaluate(
        best_model,
        test_loader,
        device,
        criterion,
    )

    labels = test_metrics[
        "labels"
    ]

    predictions = test_metrics[
        "predictions"
    ]

    print("=" * 70)
    print("Final Test Results")
    print("=" * 70)
    print()

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print()

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report = classification_report(
        labels,
        predictions,
        target_names=CLASSES,
        digits=4,
        zero_division=0,
    )

    print(
        "Classification Report:"
    )

    print(report)

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(
            range(len(CLASSES))
        ),
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    print()

    # -----------------------------------------------------
    # Save final results
    # -----------------------------------------------------

    results = {
        "model": PRETRAINED_MODEL,
        "classes": CLASSES,
        "train_samples": len(
            train_dataset
        ),
        "validation_samples": len(
            val_dataset
        ),
        "test_samples": len(
            test_dataset
        ),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "device": str(device),
        "best_validation_accuracy": (
            best_val_accuracy
        ),
        "test_accuracy": (
            test_metrics["accuracy"]
        ),
        "test_precision": (
            test_metrics["precision"]
        ),
        "test_recall": (
            test_metrics["recall"]
        ),
        "test_f1": (
            test_metrics["f1"]
        ),
        "confusion_matrix": cm.tolist(),
    }

    with open(
        RESULTS_DIR
        / "test_results.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    with open(
        RESULTS_DIR
        / "classification_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    np.savetxt(
        RESULTS_DIR
        / "confusion_matrix.csv",
        cm,
        delimiter=",",
        fmt="%d",
    )

    # -----------------------------------------------------
    # Final message
    # -----------------------------------------------------

    print("=" * 70)
    print("4-Class ViT Training Complete")
    print("=" * 70)
    print()

    print(
        f"Model saved to:"
    )

    print(
        MODEL_DIR.resolve()
    )

    print()

    print(
        f"Results saved to:"
    )

    print(
        RESULTS_DIR.resolve()
    )

    print()

    print(
        "Original 3-class model was NOT modified."
    )

    print()


if __name__ == "__main__":
    main()