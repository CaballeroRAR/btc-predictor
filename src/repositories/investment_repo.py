from datetime import datetime
from src.repositories.firestore_repo import FirestoreRepository
from src.core.schemas import InvestmentSchema

class InvestmentRepository(FirestoreRepository):
    """
    Handles persistence of user investments.
    Inherits from FirestoreRepository for low-level DB access.
    """
    def __init__(self):
        super().__init__()
        self.collection = "investments"

    def save_investment(self, investment_data):
        """
        Saves a new investment record with strict Pydantic validation.
        """
        # 1. Validation & Schema Enforcement
        if not isinstance(investment_data, InvestmentSchema):
            # Assign ID if missing before validation
            if "id" not in investment_data:
                investment_data["id"] = datetime.now().strftime("%Y%m%d%H%M%S")
            investment_data = InvestmentSchema(**investment_data)
        
        if not investment_data.id:
            investment_data.id = datetime.now().strftime("%Y%m%d%H%M%S")

        document_id = investment_data.id
        self.logger.info(f"Persisting validated investment record: {document_id}")
        return self.save(investment_data.model_dump(), self.collection, document_id)

    def get_all_investments(self):
        """Retrieve all investments and patch legacy records to match current schema version."""
        self.logger.info("Retrieving all investment records")
        raw_results = self.query(self.collection, order_by="id")
        
        patched_results = []
        for record in raw_results:
            # Legacy Patching Logic (Schema Evolution)
            if "simulation_status" not in record:
                record["simulation_status"] = "SUCCESS" # Default for old records
            if "confidence_score" not in record:
                record["confidence_score"] = 1.0 # Default for old records
            
            try:
                # Re-validate to ensure integrity
                validated = InvestmentSchema(**record)
                patched_results.append(validated.model_dump())
            except Exception as e:
                self.logger.error(f"Integrity check failed for record {record.get('id')}: {str(e)}")
        
        return patched_results

    def remove_investment(self, investment_id: str):
        """Delete an investment record."""
        self.logger.warning(f"Removing investment record: {investment_id}")
        return self.delete(self.collection, investment_id)
