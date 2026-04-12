from src.core.simulation import compute_withdrawal_plan
from src.repositories.investment_repo import InvestmentRepository
from src.utils.logger import setup_logger

logger = setup_logger("facades.simulation")

class SimulationFacade:
    """
    Coordinates investment simulations and journal persistence.
    """
    def __init__(self):
        self.investment_repo = InvestmentRepository()

    def run_investment_simulation(self, base_forecast, entry_price, profit_target_pct, investment_amt):
        """Execute simulation math."""
        logger.info(f"Running simulation: Entry ${entry_price}, Target {profit_target_pct}%")
        return compute_withdrawal_plan(base_forecast, entry_price, profit_target_pct, investment_amt)

    def save_to_journal(self, results):
        """Persist a simulation result to the secure journal."""
        logger.info("Persisting simulation to Investment Journal")
        
        # Mapping simulation keys to repository format
        data = {
            "amount": results['investment_amount'],
            "date": results['dates'][0].strftime('%Y-%m-%d'), # Simplified for journal
            "price": results['entry_price'],
            "profit_target": results['target_pct'],
            "original_withdrawal_date": results['projected_withdrawal_date'].strftime('%Y-%m-%d') if results['projected_withdrawal_date'] else None,
            "forecast_prices": results['prices'],
            "std": results['std'],
            "forecast_dates": [d.strftime('%Y-%m-%d') for d in results['dates']]
        }
        return self.investment_repo.save_investment(data)

    def get_journal_entries(self):
        """Retrieve historical investments."""
        return self.investment_repo.get_all_investments()
