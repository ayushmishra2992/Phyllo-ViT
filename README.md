# Phyllo-ViT

## Vision Transformer Based Phyllotaxy Classification and Preliminary AQI Estimation

Phyllo-ViT is a computer vision and machine learning project that analyzes leaf images to classify their **phyllotaxy pattern** and perform **preliminary Air Quality Index (AQI) estimation** using learned visual representations.

The project currently combines a **Vision Transformer (ViT)** based image-classification pipeline with a **FastAPI backend** and a **Next.js frontend**.

The long-term objective is to explore the relationship between leaf characteristics and environmental pollution indicators through image-based and multimodal analysis.

---

## Project Objectives

- Classify leaf images according to their phyllotaxy arrangement.
- Compare a traditional computer-vision baseline with a Vision Transformer.
- Extract learned visual representations from the trained ViT.
- Investigate the feasibility of estimating AQI from leaf-image representations.
- Provide a usable web interface for image-based inference.
- Establish a foundation for future multimodal integration using FTIR spectral information and pollution data.

---

## Phyllotaxy Classes

The current classification model supports three phyllotaxy classes:

- **Alternate**
- **Opposite**
- **Whorled**

> **Current limitation:** The model is trained only on these three leaf classes. A non-leaf image will therefore still be assigned to one of the available classes. A dedicated `Not a leaf` class/OOD detection mechanism is planned as a future improvement.

---

## Features

### Dataset & Image Processing

- Dataset auditing and cleaning
- Duplicate-image detection
- Image quality verification
- Stratified train/validation/test splitting
- Image preprocessing using the ViT image processor
- Exploratory data analysis

### Phyllotaxy Classification

- HOG + SVM baseline
- Pretrained Vision Transformer (ViT)
- Fine-tuned ViT classifier
- 768-dimensional CLS-token embeddings
- Confidence-based prediction output

### AQI Analysis

- Image-to-AQI dataset matching
- ViT-based image embeddings
- StandardScaler preprocessing
- Ridge Regression
- Cross-validation for model selection
- Preliminary image-to-AQI prediction

### Backend

- FastAPI inference server
- REST API for image prediction
- Health-check endpoint
- Interactive Swagger API documentation
- CPU/GPU device selection
- CUDA-supported inference when available

### Frontend

- Next.js + TypeScript
- Leaf image upload
- Image preview
- File validation
- Loading and error states
- Phyllotaxy prediction display
- Prediction confidence display
- Conditional AQI display
- Model/device information
- API integration with FastAPI
- Reset functionality

---

# Dataset

The cleaned dataset contains:

| Class | Unique Images |
|---|---:|
| Alternate | 88 |
| Opposite | 83 |
| Whorled | 94 |
| **Total** | **265** |

The original dataset contained **289 images**.

After dataset auditing:

- 0 corrupted images were identified.
- 24 exact duplicate images were identified.
- 265 unique images were retained.

### Dataset Split

A stratified split with `random_state=42` was used:

| Split | Images |
|---|---:|
| Training | 185 |
| Validation | 40 |
| Test | 40 |
| **Total** | **265** |

The held-out test set contains:

| Class | Test Images |
|---|---:|
| Alternate | 14 |
| Opposite | 12 |
| Whorled | 14 |
| **Total** | **40** |

---

# Models

## 1. HOG + SVM Baseline

A traditional computer-vision baseline was implemented using:

- Grayscale image resizing to 128 × 128
- Histogram of Oriented Gradients (HOG)
- RBF-kernel Support Vector Machine
- `C = 10`

### Test Result

| Model | Test Accuracy |
|---|---:|
| HOG + SVM | **95%** |
| ViT | **100%** |

The HOG + SVM model achieved **95% accuracy on the 40-image held-out test set**.

---

## 2. Vision Transformer

The primary classification model uses:

`google/vit-base-patch16-224-in21k`

### Training Configuration

- Architecture: Vision Transformer
- Image size: 224 × 224
- Number of classes: 3
- Batch size: 4
- Epochs: 8
- Learning rate: `2e-5`
- Optimizer: AdamW
- Weight decay: `0.01`
- Mixed precision: CUDA-supported
- Best model selected using validation accuracy

### Test Result

The ViT achieved:

- Accuracy: **100%**
- Precision: **1.00**
- Recall: **1.00**
- F1-score: **1.00**

These results are based specifically on the **40-image held-out test set** and should not be interpreted as evidence of general 100% real-world accuracy.

---

# ViT Embeddings

The trained Vision Transformer is also used as a feature extractor.

The **CLS token representation** from the ViT backbone is extracted as a learned image representation.

### Embedding Dimensions

```text
768 dimensions
