import os

# GCP Project Details
PROJECT_ID = "btc-predictor-492515"
REGION = "us-central1"
BUCKET_NAME = "btc_predictor_models"
FIRESTORE_DATABASE = "btc-pred-db"

# Image URIs
TRAINING_IMAGE_URI = f"gcr.io/{PROJECT_ID}/btc-trainer"
DASHBOARD_IMAGE_URI = f"gcr.io/{PROJECT_ID}/btc-dashboard"

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "btc_lstm_model.h5")

# Dataset Configuration
LOOKBACK_DAYS = 60
FORECAST_DAYS = 30  # Increased for "Time-to-Profit" prediction
YEARS_HISTORY = 6

# Vertex AI Training Config
MACHINE_TYPE = "n1-standard-4"
ACCELERATOR_TYPE = None
ACCELERATOR_COUNT = 0

def get_storage_path(filename, folder=DATA_DIR):
    """Utility to format GCS paths."""
    return f"gs://{BUCKET_NAME}/{folder}/{filename}"
