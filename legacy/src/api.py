from pathlib import Path

import torch
import joblib

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FOUR_CLASS_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "vit_phyllotaxy_4class"
)

THREE_CLASS_MODEL_DIR = (
    BASE_DIR
    / "models"
    / "vit_phyllotaxy"
)

AQI_MODEL_PATH = (
    BASE_DIR
    / "results"
    / "aqi_regression"
    / "aqi_ridge_model.joblib"
)


# =========================================================
# Device
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="Phyllo-ViT API",
    description=(
        "Leaf phyllotaxy classification and "
        "preliminary pollution estimation API."
    ),
    version="2.0.0",
)


# =========================================================
# Load 4-class classifier
# =========================================================

print("Loading 4-class ViT...")

four_class_processor = (
    ViTImageProcessor.from_pretrained(
        str(FOUR_CLASS_MODEL_DIR)
    )
)

four_class_model = (
    ViTForImageClassification.from_pretrained(
        str(FOUR_CLASS_MODEL_DIR)
    )
)

four_class_model.to(DEVICE)
four_class_model.eval()


# =========================================================
# Load original 3-class ViT
#
# This model is kept specifically for the AQI
# Ridge model because the Ridge model was trained
# using embeddings from this original ViT.
# =========================================================

print("Loading original 3-class ViT for AQI...")

aqi_processor = (
    ViTImageProcessor.from_pretrained(
        str(THREE_CLASS_MODEL_DIR)
    )
)

aqi_embedding_model = (
    ViTForImageClassification.from_pretrained(
        str(THREE_CLASS_MODEL_DIR)
    )
)

aqi_embedding_model.to(DEVICE)
aqi_embedding_model.eval()


# =========================================================
# Load AQI regression model
# =========================================================

print("Loading AQI Ridge model...")

aqi_model = joblib.load(
    AQI_MODEL_PATH
)


# =========================================================
# Class mapping
# =========================================================

CLASS_NAMES = [
    "Alternate",
    "Opposite",
    "Whorled",
    "Not_a_leaf",
]


# =========================================================
# Root
# =========================================================

@app.get("/")
def root():

    return {
        "project": "Phyllo-ViT",
        "status": "running",
        "message": (
            "Phyllo-ViT 4-class API is working"
        ),
        "classes": CLASS_NAMES,
        "device": str(DEVICE),
    }


# =========================================================
# Health
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "classifier": "4-class ViT",
        "classes": CLASS_NAMES,
        "aqi_model": "Ridge",
        "aqi_supported_class": "Whorled",
    }


# =========================================================
# Extract AQI-compatible embedding
# =========================================================

def extract_aqi_embedding(image):

    """
    Extract the 768-dimensional CLS embedding from
    the original 3-class ViT.

    This is intentionally separate from the new
    4-class classifier because the AQI Ridge model
    was trained on embeddings from the original
    3-class ViT.
    """

    inputs = aqi_processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = (
        inputs["pixel_values"]
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = (
            aqi_embedding_model.vit(
                pixel_values=pixel_values
            )
        )

        embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

    return embedding.cpu().numpy()


# =========================================================
# Prediction
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Validate filename
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    # -----------------------------------------------------
    # Validate image type
    # -----------------------------------------------------

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp",
    }

    if file.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Please upload a JPG, JPEG, PNG, "
                "or WEBP image."
            ),
        )

    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    try:

        contents = await file.read()

        from io import BytesIO

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unable to read image: {e}"
            ),
        )

    # -----------------------------------------------------
    # 4-class classification
    # -----------------------------------------------------

    inputs = four_class_processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = (
        inputs["pixel_values"]
        .to(DEVICE)
    )

    with torch.no_grad():

        outputs = four_class_model(
            pixel_values=pixel_values
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )

        predicted_id = (
            torch.argmax(
                probabilities,
                dim=1,
            )
            .item()
        )

        confidence = (
            probabilities[0, predicted_id]
            .item()
        )

    predicted_class = CLASS_NAMES[
        predicted_id
    ]

    # -----------------------------------------------------
    # AQI handling
    # -----------------------------------------------------

    aqi_value = None

    if predicted_class == "Whorled":

        try:

            # Use the ORIGINAL 3-class ViT
            # because the AQI Ridge model was
            # trained on those embeddings.

            aqi_embedding = (
                extract_aqi_embedding(
                    image
                )
            )

            aqi_prediction = (
                aqi_model.predict(
                    aqi_embedding
                )[0]
            )

            aqi_value = float(
                aqi_prediction
            )

            aqi_message = (
                "Preliminary AQI estimate "
                "based on the ViT embedding."
            )

        except Exception as e:

            aqi_message = (
                "AQI prediction failed: "
                f"{str(e)}"
            )

    elif predicted_class in [
        "Alternate",
        "Opposite",
    ]:

        aqi_message = (
            "AQI prediction is currently "
            "unavailable for this class because "
            "the verified image-AQI training "
            "dataset currently contains only "
            "Whorled leaf samples."
        )

    else:

        # Not_a_leaf

        aqi_message = (
            "AQI is not applicable because "
            "the uploaded image was classified "
            "as Not_a_leaf."
        )

    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return {
        "filename": file.filename,

        "phyllotaxy": {
            "class": predicted_class,
            "confidence": round(
                confidence,
                4,
            ),
        },

        "aqi": {
            "value": (
                round(
                    aqi_value,
                    2,
                )
                if aqi_value is not None
                else None
            ),
            "message": aqi_message,
        },

        "model": {
            "device": str(DEVICE),
            "classifier": "ViT 4-class",
            "embedding_dimensions": 768,
        },
    }