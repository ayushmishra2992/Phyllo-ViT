# Phyllo-ViT

## Vision Transformer Based Phyllotaxy Classification and Preliminary AQI Estimation

Phyllo-ViT is a computer vision and machine learning project that analyzes leaf images to classify their phyllotaxy pattern and perform preliminary AQI estimation using learned visual representations.

## Features

- Leaf image preprocessing and dataset analysis
- Phyllotaxy classification:
  - Alternate
  - Opposite
  - Whorled
- HOG + SVM baseline
- Pretrained Vision Transformer (ViT)
- 768-dimensional ViT embeddings
- Image-to-AQI mapping
- Preliminary AQI regression using Ridge Regression
- FastAPI backend for image inference
- CUDA/GPU-supported inference

## Results

### Phyllotaxy Classification

| Model | Test Accuracy |
|---|---:|
| HOG + SVM | 95% |
| ViT | 100% |

The ViT result is based on the 40-image held-out test set and should not be interpreted as general 100% accuracy.

### Preliminary AQI Regression

The verified image-AQI dataset contains **46 image-AQI pairs**.

| Metric | Result |
|---|---:|
| MAE | 4.36 |
| RMSE | 4.83 |
| R² | 0.822 |

The AQI experiment is preliminary because of the limited number of verified image-AQI pairs.

## System Pipeline

```text
Leaf Image
    ↓
Image Preprocessing
    ↓
ViT Phyllotaxy Classification
    ↓
768-D ViT Embedding
    ↓
Ridge Regression
    ↓
Preliminary AQI Estimate
Backend

The project uses FastAPI for model inference.

API Endpoints
GET  /
GET  /health
POST /predict

Interactive API documentation:

http://127.0.0.1:8000/docs
Tech Stack
Python
PyTorch
Hugging Face Transformers
Vision Transformer (ViT)
Scikit-learn
NumPy
Pandas
FastAPI
Uvicorn
CUDA
Project Structure
Phyllotaxy/
├── data/
├── models/
├── results/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
Current Status
Completed
Dataset audit and cleaning
Image EDA
Dataset splitting
HOG-SVM baseline
ViT classification
ViT embeddings
Image-AQI mapping
AQI regression
Ridge model selection
FastAPI backend
End-to-end image inference
Upcoming
Next.js frontend
Frontend/API integration
Final system testing
Expanded image-AQI dataset
FTIR-based multimodal integration in Phase 2
Authors

Ayush Mishra
Riya

B.Tech Computer Science and Engineering
Bennett University

Note

The AQI prediction component is a preliminary research experiment based on a limited number of verified image-AQI pairs. Further validation with a larger dataset is required before making broader environmental prediction claims.

