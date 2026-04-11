from google.cloud import firestore
import cloud_config as cloud_config
from datetime import datetime
import pandas as pd
import os
import json
from dotenv import load_dotenv

# Load local environment variables for manual migration script
load_dotenv()

COLLECTION_INVESTMENTS = "investments"
COLLECTION_PREDICTIONS = "daily_predictions"
COLLECTION_LIVE_DRIFT = "live_drift_audit"

class DatabaseManager:
    def __init__(self):
        # Retrieve custom database ID if provided, otherwise fallback to (default)
        db_id = os.getenv("FIRESTORE_DATABASE", "(default)")
        self.db = firestore.Client(project=cloud_config.PROJECT_ID, database=db_id)
        self.investments_col = COLLECTION_INVESTMENTS
        self.predictions_col = COLLECTION_PREDICTIONS
        self.drift_col = COLLECTION_LIVE_DRIFT

    def log_live_drift(self, forecast_date, prediction, market_price, drift_pct, source="USER"):
        """Log high-frequency tactical drift snapshots to a dedicated collection."""
        try:
            doc_ref = self.db.collection(self.drift_col).document()
            doc_ref.set({
                'timestamp': firestore.SERVER_TIMESTAMP,
                'forecast_date': forecast_date, 
                'predicted_price': float(prediction),
                'market_price': float(market_price),
                'drift_pct': float(drift_pct),
                'source': source
            })
            return True
        except Exception as e:
            print(f"Error logging live drift: {str(e)}")
            return False

    # --- Investments ---
    
    def save_investment(self, investment_data):
        """Save or update an investment record."""
        doc_id = investment_data.get("id", datetime.now().strftime("%Y%m%d%H%M%S"))
        self.db.collection(self.investments_col).document(doc_id).set(investment_data)
        return doc_id

    def get_investments(self):
        """Retrieve all investments sorted by date."""
        docs = self.db.collection(self.investments_col).order_by("date").stream()
        return [doc.to_dict() for doc in docs]

    def delete_investment(self, inv_id):
        """Remove an investment."""
        self.db.collection(self.investments_col).document(str(inv_id)).delete()

    def get_live_drift_history(self, days=30):
        """Retrieve recent SYSTEM-only drift logs for model training enrichment."""
        try:
            cutoff = datetime.now() - pd.Timedelta(days=days)
            docs = self.db.collection(self.drift_col)\
                .where("source", "==", "SYSTEM")\
                .where("timestamp", ">=", cutoff)\
                .order_by("timestamp")\
                .stream()
            
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            print(f"Error fetching drift history: {str(e)}")
            return []

    # --- Predictions & Evaluation ---

    def log_prediction(self, sim_run_date, forecast_date, predicted_price, actual_price=None):
        """
        Log a prediction. 
        Document ID is a combination of sim_run_date and forecast_date to handle multiple simulations.
        """
        doc_id = f"{sim_run_date}_{forecast_date}"
        data = {
            "sim_run_date": sim_run_date,
            "forecast_date": forecast_date,
            "predicted_price": float(predicted_price),
            "actual_price": float(actual_price) if actual_price is not None else None,
            "timestamp": firestore.SERVER_TIMESTAMP
        }
        self.db.collection(self.predictions_col).document(doc_id).set(data, merge=True)

    def get_prediction_history(self, limit=100):
        """Retrieve historical predictions for evaluation."""
        docs = self.db.collection(self.predictions_col).order_by("forecast_date", direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [doc.to_dict() for doc in docs]

    def update_actual_price(self, forecast_date, actual_price):
        """Update the actual price for all predictions targeting a specific date."""
        # Find all documents where forecast_date matches and actual_price is None
        query = self.db.collection(self.predictions_col).where("forecast_date", "==", forecast_date)
        docs = query.stream()
        
        batch = self.db.batch()
        count = 0
        for doc in docs:
            batch.update(doc.reference, {"actual_price": float(actual_price)})
            count += 1
        
        if count > 0:
            batch.commit()
        return count

    # --- Dashboard Snapshots ---

    def save_system_snapshot(self, snapshot_data):
        """
        Store a snapshot of the current dashboard analysis (forecast, drift, impacts).
        Keyed by 'latest' for quick retrieval, but includes a timestamp.
        """
        snapshot_data["timestamp"] = firestore.SERVER_TIMESTAMP
        self.db.collection("system_state").document("latest_snapshot").set(snapshot_data)

    def get_latest_snapshot(self):
        """Retrieve the most recent dashboard snapshot."""
        doc = self.db.collection("system_state").document("latest_snapshot").get()
        if doc.exists:
            return doc.to_dict()
        return None

    # --- Migration Helpers ---

    def migrate_json_investments(self, file_path="data/investments.json"):
        """Migrate local investments.json to Firestore."""
        if not os.path.exists(file_path):
            return 0
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        count = 0
        for entry in data:
            self.save_investment(entry)
            count += 1
        return count

    def migrate_csv_predictions(self, file_path="data/prediction_history.csv"):
        """Migrate local prediction_history.csv to Firestore."""
        if not os.path.exists(file_path):
            return 0
        df = pd.read_csv(file_path)
        count = 0
        for _, row in df.iterrows():
            # CSV columns: timestamp, sim_run_date, forecast_date, predicted_price
            self.log_prediction(
                sim_run_date=row['sim_run_date'],
                forecast_date=row['forecast_date'],
                predicted_price=row['predicted_price']
            )
            count += 1
        return count

if __name__ == "__main__":
    # Quick test/migration
    db = DatabaseManager()
    print("Migrating investments...")
    i_count = db.migrate_json_investments()
    print(f"Migrated {i_count} investments.")
    
    print("Migrating predictions...")
    p_count = db.migrate_csv_predictions()
    print(f"Migrated {p_count} predictions.")
