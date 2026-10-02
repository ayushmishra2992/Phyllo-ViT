import io
import os
import joblib
import numpy as np
import torch

from PIL import Image

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from transformers import (
    ViTImageProcessor,
    ViTForImageClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

VIT_MODEL_PATH = "models/vit_phyllotaxy"

AQI_MODEL_PATH = (
    "results/aqi_regression/"
    "aqi_ridge_model.joblib"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Phyllo-ViT API",
    description=(
        "Leaf phyllotaxy classification and "
        "preliminary AQI prediction API"
    ),
    version="1.0.0"
)


# ============================================================
# LOAD ViT MODEL
# ============================================================

print("=" * 70)
print("LOADING PHYLLO-ViT MODEL")
print("=" * 70)

print("Device:", DEVICE)

processor = ViTImageProcessor.from_pretrained(
    VIT_MODEL_PATH
)

model = ViTForImageClassification.from_pretrained(
    VIT_MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("ViT model loaded successfully.")

print("Class mapping:")
print(model.config.id2label)


# ============================================================
# LOAD AQI RIDGE MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING AQI REGRESSION MODEL")
print("=" * 70)

aqi_model = joblib.load(
    AQI_MODEL_PATH
)

print("AQI Ridge model loaded successfully.")


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "Phyllo-ViT",
        "status": "running",
        "device": str(DEVICE),
        "models": {
            "vit_classifier": True,
            "aqi_regressor": True
        },
        "classes": [
            "Alternate",
            "Opposite",
            "Whorled"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "vit_model_loaded": True,
        "aqi_model_loaded": True,
        "device": str(DEVICE)
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if file.content_type is not None:

        if not file.content_type.startswith("image/"):

            raise HTTPException(
                status_code=400,
                detail="Uploaded file must be an image."
            )

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")


        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }


        # ====================================================
        # ViT INFERENCE
        # ====================================================

        with torch.no_grad():

            outputs = model(**inputs)

            logits = outputs.logits

            probabilities = torch.softmax(
                logits,
                dim=-1
            )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        predicted_id = int(
            torch.argmax(
                probabilities,
                dim=-1
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                predicted_id
            ].item()
        )

        predicted_class = (
            model.config.id2label[
                predicted_id
            ]
        )


        # ====================================================
        # EXTRACT ViT CLS EMBEDDING
        # ====================================================

        with torch.no_grad():

            vit_outputs = model.vit(
                **inputs
            )

            cls_embedding = (
                vit_outputs
                .last_hidden_state[:, 0, :]
            )

        embedding = (
            cls_embedding
            .detach()
            .cpu()
            .numpy()
        )


        # ====================================================
        # AQI PREDICTION
        # ====================================================

        predicted_aqi = None

        aqi_message = (
            "AQI prediction is currently "
            "available only for the Whorled class "
            "because the verified image-AQI training "
            "dataset contains Whorled leaf samples."
        )


        if predicted_class == "Whorled":

            predicted_aqi = float(
                aqi_model.predict(
                    embedding
                )[0]
            )

            # Keep AQI within a sensible display range.
            predicted_aqi = round(
                predicted_aqi,
                2
            )

            aqi_message = (
                "Preliminary AQI estimate based on "
                "the ViT embedding."
            )


        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "filename": file.filename,

            "phyllotaxy": {
                "class": predicted_class,
                "confidence": round(
                    confidence,
                    4
                )
            },

            "aqi": {
                "value": predicted_aqi,
                "message": aqi_message
            },

            "model": {
                "device": str(DEVICE),
                "embedding_dimensions": int(
                    embedding.shape[1]
                )
            }
        }


    except HTTPException:
        raise


    except Exception as error:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to process image: "
                f"{str(error)}"
            )
        )