# Cloud Run Streamlit App Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (for Plotly/Streamlit)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install project requirements
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy source code and assets
COPY src/ /app/src/
COPY models/ /app/models/
COPY .streamlit/ /app/.streamlit/

# Streamlit-specific config
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

# Entry point for Cloud Run
EXPOSE 8080
CMD ["streamlit", "run", "src/dashboard.py"]
