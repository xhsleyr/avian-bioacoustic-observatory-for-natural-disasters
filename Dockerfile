# ==============================================================================
# PROJECT INTERA — Planetary Bioacoustic Intelligence Engine
# Multi-stage production container for Streamlit UI + FastAPI Headless Engine
# ==============================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

# Install system audio dependencies for librosa & soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn pytest

# Copy application source and assets
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .
COPY .streamlit/ ./.streamlit/

EXPOSE 8501 8000

# Default: Streamlit UI
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
