# Vertex AI Training Dockerfile (CPU Optimized)
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy source code (cloud_config is inside src/)
COPY src/ /app/src/

# Entry point for training
CMD ["python", "src/main_trainer.py"]
