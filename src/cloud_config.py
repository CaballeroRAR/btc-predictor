import os

# GCP Project Details
PROJECT_ID = "btc-predictor-492515"
REGION = "us-central1"
BUCKET_NAME = "btc-predictor-492515_cloudbuild"

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "btc_lstm_model.h5")

# Dataset Configuration
LOOKBACK_DAYS = 60
FORECAST_DAYS = 30  # Multi-step prediction window
FEATURE_COUNT = 12  # Macro Gravity Schema (OHLCV, 4 asset ratios, RSI, Sentiment, Trends)
YEARS_HISTORY = 6

# Default Baselines
TRENDS_BASELINE = 50.0  # Fallback for Google Trends API lockout
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml"
]

# Vertex AI Training Config
MACHINE_TYPE = "n1-standard-4"
ACCELERATOR_TYPE = "NVIDIA_TESLA_T4"
ACCELERATOR_COUNT = 1

def get_storage_path(filename, folder=DATA_DIR):
    """Utility to format GCS paths."""
    return f"gs://{BUCKET_NAME}/{folder}/{filename}"
