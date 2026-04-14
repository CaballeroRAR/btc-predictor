import os
from datetime import datetime
from src import cloud_config
from src.core.data_orchestrator import data_orchestrator
from src.repositories.asset_repo import AssetRepository
from src.utils.logger import setup_logger
from src import vertex_trigger as vertex

logger = setup_logger("facades.lifecycle")

class LifecycleFacade:
    """
    Orchestrates high-level model lifecycle operations:
    - Status & Age reporting
    - Manual/Auto asset synchronization from Cloud Storage
    - Triggering training jobs on Vertex AI
    """
    def __init__(self):
        self.assets = AssetRepository()

    def get_system_status(self):
        """Returns health metrics and model freshness."""
        if not os.path.exists(cloud_config.MODEL_PATH):
            return {"status": "ERROR", "message": "Model files missing locally"}
            
        mtime = os.path.getmtime(cloud_config.MODEL_PATH)
        dt_updated = datetime.fromtimestamp(mtime)
        age_days = (datetime.now().date() - dt_updated.date()).days
        
        return {
            "status": "OPERATIONAL",
            "model_age_days": age_days,
            "last_training_date": dt_updated,
            "paths": {
                "model": cloud_config.MODEL_PATH,
                "scaler": cloud_config.SCALER_PATH
            }
        }

    def load_model_assets(self):
        """
        Loads and returns the LSTM model and scaler.
        Performs a dynamic GCS sync if artifacts are missing locally.
        """
        if not os.path.exists(cloud_config.MODEL_PATH) or not os.path.exists(cloud_config.SCALER_PATH):
            logger.info("Critical artifacts missing. Initiating automatic cloud synchronization...")
            self.sync_assets(force=True)
            
        model = self.assets.load_model("btc_lstm_model.h5")
        scaler = self.assets.load_scaler("scaler.pkl")
        return model, scaler

    def load_dataset(self, force=False):
        """Proxy to DataOrchestrator for dataset retrieval."""
        return data_orchestrator.prepare_dataset(force_refresh=force)

    def sync_assets(self, force=False):
        """Synchronizes model and scaler from GCP."""
        logger.info(f"Triggering asset sync (Force={force})")
        # Currently, asset_repo has sync_from_cloud. We use it.
        # Note: asset_repo needs to handle both files.
        try:
            self.assets.sync_from_cloud("btc_lstm_model.h5")
            self.assets.sync_from_cloud("scaler.pkl")
            return True
        except Exception as e:
            logger.error(f"Sync failed: {str(e)}")
            return False

    def publish_assets(self):
        """Pushes local assets to GCP (Used by Trainer)."""
        logger.info("Initiating cloud publishing of local assets...")
        try:
            self.assets.sync_to_cloud("btc_lstm_model.h5")
            self.assets.sync_to_cloud("scaler.pkl")
            return True
        except Exception as e:
            logger.error(f"Publishing failed: {str(e)}")
            return False

    def launch_retraining(self):
        """Spawns a Cloud Training job."""
        logger.warning("Initiating Vertex AI Training Trigger")
        try:
            job = vertex.trigger_training_job()
            return {"status": "success", "job_id": job.resource_name}
        except Exception as e:
            logger.error(f"Training trigger failed: {str(e)}")
            raise e

    def get_active_training_jobs(self, limit=1):
        """Returns telemetry for recent Cloud training jobs."""
        jobs = vertex.get_latest_training_jobs(limit=limit)
        if not jobs:
            return None
            
        job = jobs[0]
        return {
            "id": job.name,
            "status": vertex.get_status_summary(job),
            "raw_job": job
        }
