from datetime import datetime
from src.repositories.firestore_repo import FirestoreRepository

class InvestmentRepository(FirestoreRepository):
    """
    Handles persistence of user investments.
    Inherits from FirestoreRepository for low-level DB access.
    """
    def __init__(self):
        super().__init__()
        self.collection = "investments"

    def save_investment(self, investment_data: dict):
        """
        Saves a new investment record.
        Ensures a unique ID based on timestamp if not provided.
        """
        if "id" not in investment_data:
            investment_data["id"] = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Ensure serialization hygiene
        for key in ["forecast_prices", "calibrated_prices", "std"]:
            if key in investment_data and hasattr(investment_data[key], "tolist"):
                investment_data[key] = investment_data[key].tolist()

        document_id = investment_data["id"]
        self.logger.info(f"Persisting investment record: {document_id}")
        return self.save(investment_data, self.collection, document_id)

    def get_all_investments(self):
        """Retrieve all investments ordered by timestamp."""
        self.logger.info("Retrieving all investment records")
        return self.query(self.collection, order_by="date")

    def remove_investment(self, investment_id: str):
        """Delete an investment record."""
        self.logger.warning(f"Removing investment record: {investment_id}")
        return self.delete(self.collection, investment_id)
