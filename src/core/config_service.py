import os
from dotenv import load_dotenv
from threading import Lock

class ConfigService:
    """
    Singleton Configuration Service.
    Enforces a single point of truth for project-wide settings.
    Supports environment variable overrides for CI/CD and deployment.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigService, cls).__new__(cls)
                cls._instance._init_config()
            return cls._instance

    def _init_config(self):
        # 1. Load environment from .env file
        load_dotenv()

        # 2. GCP Project details (Mandatory)
        self.PROJECT_ID = os.getenv("PROJECT_ID")
        if not self.PROJECT_ID:
            raise EnvironmentError(
                "CRITICAL: PROJECT_ID environment variable is missing. "
                "Please define it in your .env file or shell environment."
            )

        self.REGION = os.getenv("REGION", "us-central1")
        self.BUCKET_NAME = os.getenv("BUCKET_NAME", "btc_predictor_models")
        self.FIRESTORE_DATABASE = os.getenv("FIRESTORE_DATABASE", "btc-pred-db")

        # Image URIs
        self.TRAINING_IMAGE_URI = os.getenv("TRAINING_IMAGE_URI", f"gcr.io/{self.PROJECT_ID}/btc-trainer")
        self.DASHBOARD_IMAGE_URI = os.getenv("DASHBOARD_IMAGE_URI", f"gcr.io/{self.PROJECT_ID}/btc-dashboard")

        # Paths
        self.DATA_DIR = os.getenv("DATA_DIR", "data")
        self.MODEL_DIR = os.getenv("MODEL_DIR", "models")
        self.SCALER_PATH = os.path.join(self.MODEL_DIR, "scaler.pkl")
        self.MODEL_PATH = os.path.join(self.MODEL_DIR, "btc_lstm_model.h5")

        # Dataset Configuration
        self.LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", 60))
        self.FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", 30))
        self.YEARS_HISTORY = int(os.getenv("YEARS_HISTORY", 6))

        # Vertex AI Training Config
        self.MACHINE_TYPE = os.getenv("MACHINE_TYPE", "n1-standard-4")
        self.ACCELERATOR_TYPE = os.getenv("ACCELERATOR_TYPE", None)
        self.ACCELERATOR_COUNT = int(os.getenv("ACCELERATOR_COUNT", 0))

    def get_storage_path(self, filename, folder=None):
        """Utility to format GCS paths."""
        target_folder = folder or self.DATA_DIR
        return f"gs://{self.BUCKET_NAME}/{target_folder}/{filename}"

# Global accessor for the singleton
config_service = ConfigService()
