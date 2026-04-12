from datetime import datetime
from src.repositories.firestore_repo import FirestoreRepository

class CalibrationRepository(FirestoreRepository):
    """
    Handles persistence of sentiment calibration (market alignment) state.
    """
    def __init__(self):
        super().__init__()
        self.collection = "calibration_state"
        self.latest_doc = "latest"

    def save_state(self, drift_value: float, reference_price: float, model_path: str):
        """Save the latest market alignment state."""
        state = {
            "last_calibration_date": str(datetime.now()),
            "drift_value": float(drift_value),
            "reference_price": float(reference_price),
            "model_path": model_path
        }
        self.logger.info(f"Saving new calibration state: {drift_value:+.2f} drift")
        return self.save(state, self.collection, self.latest_doc)

    def load_latest_state(self):
        """Retrieve the latest drift value."""
        self.logger.info("Loading latest calibration state")
        return self.get(self.collection, self.latest_doc)
