from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize

from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split"
    / "split_metadata.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "baseline"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "Alternate",
    "Opposite",
    "Whorled"
]

# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(SPLIT_METADATA)

print("=" * 60)
print("PHYLLO-ViT BASELINE MODEL")
print("=" * 60)

print(f"\nTotal images: {len(df)}")

# ============================================================
# IMAGE → HOG FEATURES
# ============================================================

def extract_hog(image_path):

    image = Image.open(
        PROJECT_ROOT / image_path
    ).convert("RGB")

    image = np.array(image)

    # Resize
    image = resize(
        image,
        IMAGE_SIZE,
        anti_aliasing=True
    )

    # Convert to grayscale
    gray = rgb2gray(image)

    # HOG feature extraction
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


def build_features(dataframe):

    X = []
    y = []

    total = len(dataframe)

    for index, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1
    ):

        features = extract_hog(
            row["image_path"]
        )

        X.append(features)
        y.append(row["phyllotaxy"])

        if index % 25 == 0 or index == total:
            print(
                f"Processed {index}/{total} images"
            )

    return np.array(X), np.array(y)


# ============================================================
# CREATE FEATURES
# ============================================================

print("\nExtracting HOG features...")

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]

X_train, y_train = build_features(train_df)
X_val, y_val = build_features(val_df)
X_test, y_test = build_features(test_df)

print("\nFeature shapes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ============================================================
# TRAIN SVM
# ============================================================

print("\nTraining SVM baseline...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 60)
print("BASELINE RESULTS")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=CLASS_NAMES,
        zero_division=0
    )
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=CLASS_NAMES
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title("Baseline SVM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES,
    rotation=20
)

plt.yticks(
    range(len(CLASS_NAMES)),
    CLASS_NAMES
)

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["HOG + SVM"],
    "Accuracy": [accuracy],
    "Precision": [precision],
    "Recall": [recall],
    "F1": [f1]
})

results.to_csv(
    RESULTS_DIR / "baseline_metrics.csv",
    index=False
)

# ============================================================
# SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    model,
    RESULTS_DIR / "baseline_svm.joblib"
)

print("\nSaved:")
print(
    RESULTS_DIR / "baseline_metrics.csv"
)

print(
    RESULTS_DIR / "confusion_matrix.png"
)

print(
    RESULTS_DIR / "baseline_svm.joblib"
)

print("\nBaseline model completed successfully.")

print("=" * 60)