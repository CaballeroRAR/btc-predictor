import os
from datetime import datetime
from src.repositories.asset_repo import AssetRepository
from src.utils.logger import setup_logger
import cloud_config as cloud_config
import vertex_trigger as vertex

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

    def launch_retraining(self):
        """Spawns a Cloud Training job."""
        logger.warning("Initiating Vertex AI Training Trigger")
        try:
            job = vertex.trigger_training_job()
            return {"status": "success", "job_id": job.resource_name}
        except Exception as e:
            logger.error(f"Training trigger failed: {str(e)}")
            raise e
