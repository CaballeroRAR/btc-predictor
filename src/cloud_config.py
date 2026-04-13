from src.core.config_service import config_service

# Industrial Compatibility Layer
# All global constants are now proxied via the ConfigService singleton.

PROJECT_ID = config_service.PROJECT_ID
REGION = config_service.REGION
BUCKET_NAME = config_service.BUCKET_NAME
FIRESTORE_DATABASE = config_service.FIRESTORE_DATABASE

TRAINING_IMAGE_URI = config_service.TRAINING_IMAGE_URI
DASHBOARD_IMAGE_URI = config_service.DASHBOARD_IMAGE_URI

DATA_DIR = config_service.DATA_DIR
MODEL_DIR = config_service.MODEL_DIR
SCALER_PATH = config_service.SCALER_PATH
MODEL_PATH = config_service.MODEL_PATH

LOOKBACK_DAYS = config_service.LOOKBACK_DAYS
FORECAST_DAYS = config_service.FORECAST_DAYS
YEARS_HISTORY = config_service.YEARS_HISTORY

MACHINE_TYPE = config_service.MACHINE_TYPE
ACCELERATOR_TYPE = config_service.ACCELERATOR_TYPE
ACCELERATOR_COUNT = config_service.ACCELERATOR_COUNT

def get_storage_path(filename, folder=None):
    """Refactored Proxy to ConfigService."""
    return config_service.get_storage_path(filename, folder)
