# Backend image (FastAPI + both ViT models, CPU). Works on Hugging Face Spaces (Docker), Railway,
# Fly.io, Google Cloud Run, Azure, any VM.  Needs ~2 GB RAM (or ~1.2 GB with PHYLLO_LOAD_AQI_MODEL=0).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PORT=7860 PHYLLO_LOAD_AQI_MODEL=1

WORKDIR /app
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY phyllo ./phyllo
COPY api ./api
COPY models ./models
COPY artifacts ./artifacts
COPY results ./results
COPY legacy/results ./legacy/results

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 app && chown -R app /app
USER app
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
