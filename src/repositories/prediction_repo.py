from datetime import datetime
from src.repositories.firestore_repo import FirestoreRepository
from src.core.schemas import PredictionSchema
import pandas as pd

class PredictionRepository(FirestoreRepository):
    """
    Handles logging of model predictions and matching them with market actuals.
    """
    def __init__(self):
        super().__init__()
        self.collection = "daily_predictions"

    def log_prediction_batch(self, forecast_dates, predicted_prices):
        """Append a batch of new predictions with Pydantic validation."""
        sim_run_date = datetime.now().strftime("%Y%m%d")
        
        for d, p in zip(forecast_dates, predicted_prices):
            forecast_date_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
            document_id = f"{sim_run_date}_{forecast_date_str}"
            
            # Instantiate Schema (Auto-validates types)
            prediction = PredictionSchema(
                sim_run_date=sim_run_date,
                forecast_date=forecast_date_str,
                predicted_price=float(p)
            )
            
            self.save(prediction.model_dump(), self.collection, document_id)
        
        self.logger.info(f"Logged {len(forecast_dates)} validated predictions for run {sim_run_date}")

    def update_actual_price_matching(self, forecast_date: str, actual_price: float):
        """Update all prediction records targeting a specific date with its actual closing price."""
        self.logger.info(f"Syncing actual price ${actual_price} for {forecast_date}")
        
        # Query for all documents where forecast_date matches
        docs = self.db.collection(self.collection).where("forecast_date", "==", forecast_date).stream()
        
        batch = self.db.batch()
        count = 0
        for doc in docs:
            batch.update(doc.reference, {"actual_price": float(actual_price)})
            count += 1
        
        if count > 0:
            batch.commit()
            self.logger.info(f"Matched actual price in {count} records")
        return count

    def get_history(self, limit=200):
        """Fetch historical predictions for performance evaluation."""
        self.logger.info(f"Retrieving prediction history (limit: {limit})")
        return self.query(self.collection, order_by="forecast_date", limit=limit)
