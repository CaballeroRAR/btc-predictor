from abc import ABC, abstractmethod
import os
import json
from datetime import datetime
from src.utils.logger import setup_logger
from src import cloud_config

class BaseRepository(ABC):
    """
    Abstract Base Class for the Repository Pattern.
    Ensures a consistent interface for all persistence layers.
    Includes a Resilience Layer with local filesystem fallback.
    """
    def __init__(self, name: str):
        self.logger = setup_logger(name)
        self.resilience_dir = os.path.join(cloud_config.DATA_DIR, "resilience")
        self.logger.info(f"Initializing {name} (Resilience Dir: {self.resilience_dir})")

    @abstractmethod
    def save(self, data: dict, collection: str, document_id: str = None):
        pass

    @abstractmethod
    def get(self, collection: str, document_id: str):
        pass

    @abstractmethod
    def delete(self, collection: str, document_id: str):
        pass

    def _save_local(self, data: dict, collection: str, document_id: str):
        """Fallback: Save data locally as JSON if cloud sync fails."""
        try:
            target_dir = os.path.join(self.resilience_dir, collection)
            os.makedirs(target_dir, exist_ok=True)
            
            # Use timestamp to avoid collision for auto-gen IDs
            id_str = document_id or f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            path = os.path.join(target_dir, f"{id_str}.json")
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=4, default=str)
            
            self.logger.warning(f"RESILIENCE: Data saved locally to {path}")
            return id_str
        except Exception as e:
            self.logger.error(f"RESILIENCE CRITICAL: Local save failed: {str(e)}")
            return None

    def _get_local(self, collection: str, document_id: str):
        """Fallback: Retrieve data locally if cloud is unreachable."""
        try:
            path = os.path.join(self.resilience_dir, collection, f"{document_id}.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"RESILIENCE: Data recovered from local file {path}")
                return data
        except Exception as e:
            self.logger.error(f"RESILIENCE: Local recovery failed: {str(e)}")
        return None
