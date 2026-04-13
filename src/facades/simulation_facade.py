from src.core.simulation_orchestrator import SimulationOrchestrator
from src.repositories.investment_repo import InvestmentRepository
from src.utils.logger import setup_logger

logger = setup_logger("facades.simulation")

class SimulationFacade:
    """
    Coordinates investment simulations and journal persistence.
    """
    def __init__(self):
        self.investment_repo = InvestmentRepository()
        self.orchestrator = SimulationOrchestrator()

    def run_investment_simulation(self, base_forecast, entry_price, profit_target_pct, investment_amt):
        """Execute simulation via domain orchestrator."""
        logger.info(f"Delegating simulation to Orchestrator: Entry ${entry_price}")
        return self.orchestrator.run_simulation(
            base_forecast, entry_price, profit_target_pct, investment_amt
        )

    def save_to_journal(self, validated_result):
        """Persist a validated simulation result to the secure journal."""
        logger.info("Persisting validated simulation to Investment Journal")
        return self.investment_repo.save_investment(validated_result)

    def get_journal_entries(self):
        """Retrieve historical investments."""
        return self.investment_repo.get_all_investments()

    def delete_entry(self, inv_id):
        """Remove an investment record from the journal."""
        logger.warning(f"Deleting investment record: {inv_id}")
        return self.investment_repo.remove_investment(inv_id)
