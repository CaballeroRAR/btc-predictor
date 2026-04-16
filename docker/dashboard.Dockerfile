# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies to a local directory
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Install runtime dependencies (e.g., libgomp for lightgbm/tensorflow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy the rest of the application code
COPY . .

# Expose the standard Streamlit port
EXPOSE 8080

# Command to run the Streamlit dashboard
CMD ["streamlit", "run", "src/main_dashboard.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
