import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor, ViTForImageClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "split")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "vit")
MODEL_DIR = os.path.join(BASE_DIR, "models", "vit_phyllotaxy")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 2e-5

MODEL_NAME = "google/vit-base-patch16-224-in21k"


# ============================================================
# GPU INFORMATION
# ============================================================

print("=" * 60)
print("Phyllo-ViT — Vision Transformer Training")
print("=" * 60)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"VRAM: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print(f"Model: {MODEL_NAME}")
print("=" * 60)


# ============================================================
# IMAGE PROCESSOR
# ============================================================

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)


def vit_transform(image):
    """
    Converts a PIL image into the tensor format expected by ViT.
    """
    processed = processor(images=image, return_tensors="pt")
    return processed["pixel_values"].squeeze(0)


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = ImageFolder(
    TRAIN_DIR,
    transform=vit_transform
)

val_dataset = ImageFolder(
    VAL_DIR,
    transform=vit_transform
)

test_dataset = ImageFolder(
    TEST_DIR,
    transform=vit_transform
)

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images:       {len(test_dataset)}")

print("\nClasses:")
print(train_dataset.class_to_idx)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

id2label = {
    0: "Alternate",
    1: "Opposite",
    2: "Whorled"
}

label2id = {
    "Alternate": 0,
    "Opposite": 1,
    "Whorled": 2
}

print("\nLoading pretrained ViT...")

model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

model.to(DEVICE)

print("Model loaded successfully.")


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()

if use_amp:
    scaler = torch.amp.GradScaler("cuda")
else:
    scaler = None


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs.logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(pixel_values=images)

            loss = criterion(outputs.logits, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    loss = running_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return loss, accuracy, all_labels, all_predictions


# ============================================================
# TRAINING LOOP
# ============================================================

print("\nStarting training...")
print("=" * 60)

best_val_accuracy = 0.0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


for epoch in range(EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy, _, _ = evaluate(
        val_loader
    )

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        model.save_pretrained(MODEL_DIR)
        processor.save_pretrained(MODEL_DIR)

        print(
            f"  ✓ Best model saved "
            f"(validation accuracy: {val_accuracy:.4f})"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)

with open(history_path, "w") as f:
    json.dump(history, f, indent=4)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model = ViTForImageClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating on held-out test set...")

test_loss, test_accuracy, test_labels, test_predictions = evaluate(
    test_loader
)

precision, recall, f1, _ = precision_recall_fscore_support(
    test_labels,
    test_predictions,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("FINAL ViT TEST RESULTS")
print("=" * 60)

print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"F1 Score      : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,
    target_names=[
        "Alternate",
        "Opposite",
        "Whorled"
    ],
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    RESULTS_DIR,
    "vit_metrics.txt"
)

with open(metrics_path, "w") as f:

    f.write("Phyllo-ViT Classification Results\n")
    f.write("=" * 50 + "\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write(f"Weighted F1: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("ViT Phyllotaxy Classification — Confusion Matrix")

plt.xticks(
    range(3),
    ["Alternate", "Opposite", "Whorled"]
)

plt.yticks(
    range(3),
    ["Alternate", "Opposite", "Whorled"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(cm_path, dpi=300)

plt.close()


# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure(figsize=(7, 5))

plt.plot(
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("ViT Training and Validation Accuracy")

plt.legend()

plt.tight_layout()

accuracy_curve_path = os.path.join(
    RESULTS_DIR,
    "accuracy_curve.png"
)

plt.savefig(
    accuracy_curve_path,
    dpi=300
)

plt.close()


plt.figure(figsize=(7, 5))

plt.plot(
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("ViT Training and Validation Loss")

plt.legend()

plt.tight_layout()

loss_curve_path = os.path.join(
    RESULTS_DIR,
    "loss_curve.png"
)

plt.savefig(
    loss_curve_path,
    dpi=300
)

plt.close()


print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Best model: {MODEL_DIR}")
print(f"Results:    {RESULTS_DIR}")
print(f"Test accuracy: {test_accuracy:.4f}")