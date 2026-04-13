import os
import pickle
import joblib
import keras
import hashlib
from google.cloud import storage
from src.repositories.base import BaseRepository
import src.cloud_config as cloud_config

class AssetRepository(BaseRepository):
    """
    Manages local and cloud-based assets (.h5 models, .pkl scalers).
    Handles synchronization between GCS and the local runtime environment.
    """
    def __init__(self):
        super().__init__("repositories.assets")
        self.storage_client = storage.Client(project=cloud_config.PROJECT_ID)
        self.bucket = self.storage_client.bucket(cloud_config.BUCKET_NAME)

    def save_model(self, model, filename):
        """Save a Keras model locally."""
        path = os.path.join(cloud_config.MODEL_DIR, filename)
        os.makedirs(cloud_config.MODEL_DIR, exist_ok=True)
        model.save(path)
        self.logger.info(f"Model saved locally to {path}")

    def load_model(self, filename, expected_hash=None):
        """Load a Keras model locally with optional integrity check."""
        path = os.path.join(cloud_config.MODEL_DIR, filename)
        if os.path.exists(path):
            if expected_hash:
                actual_hash = self.calculate_hash(path)
                if actual_hash != expected_hash:
                    self.logger.error(f"CORRUPTION DETECTED: {filename} hash mismatch!")
                    self.logger.error(f"Expected: {expected_hash} | Actual: {actual_hash}")
                    return None
            
            # compile=False avoids Keras 3 deserialization issues for h5 files
            model = keras.models.load_model(path, compile=False)
            self.logger.info(f"Model loaded from {path} (Integrity Verified)")
            return model
        self.logger.error(f"Model file not found: {path}")
        return None

    def save_scaler(self, scaler, filename):
        """Save a Scikit-learn scaler locally using joblib."""
        path = os.path.join(cloud_config.MODEL_DIR, filename)
        os.makedirs(cloud_config.MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, path)
        self.logger.info(f"Scaler saved locally to {path} (joblib)")

    def load_scaler(self, filename):
        """Load a Scikit-learn scaler locally using joblib."""
        path = os.path.join(cloud_config.MODEL_DIR, filename)
        if os.path.exists(path):
            scaler = joblib.load(path)
            self.logger.info(f"Scaler loaded from {path} (joblib)")
            return scaler
        self.logger.error(f"Scaler file not found: {path}")
        return None

    def sync_from_cloud(self, filename):
        """Pull an asset from GCS to the local environment."""
        self.logger.info(f"Syncing {filename} from gs://{cloud_config.BUCKET_NAME}")
        blob = self.bucket.blob(f"{cloud_config.MODEL_DIR}/{filename}")
        local_path = os.path.join(cloud_config.MODEL_DIR, filename)
        os.makedirs(cloud_config.MODEL_DIR, exist_ok=True)
        blob.download_to_filename(local_path)
        self.logger.info(f"Successfully synced {filename}")

    def sync_to_cloud(self, filename):
        """Push a local asset to GCS."""
        local_path = os.path.join(cloud_config.MODEL_DIR, filename)
        if not os.path.exists(local_path):
            self.logger.error(f"Cannot sync to cloud: Local file missing at {local_path}")
            return False
            
        self.logger.info(f"Uploading {filename} to gs://{cloud_config.BUCKET_NAME}")
        blob = self.bucket.blob(f"{cloud_config.MODEL_DIR}/{filename}")
        blob.upload_from_filename(local_path)
        self.logger.info(f"Successfully uploaded {filename} to GCS")
        return True

    def save(self, data, target):
        """Generic override from BaseRepository."""
        pass # Specific methods preferred for models/scalers

    def get(self, target):
        """Generic override from BaseRepository."""
        pass

    def delete(self, target):
        """Generic override from BaseRepository."""
        path = os.path.join(cloud_config.MODEL_DIR, target)
        if os.path.exists(path):
            os.remove(path)
            self.logger.warning(f"Deleted local asset: {path}")
            return True
        return False

    def calculate_hash(self, path):
        """Compute SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Hash calculation failed: {str(e)}")
            return ""
