import numpy as np
import pandas as pd
from datetime import datetime
from src.core.simulation import find_target_crossing_date
from src.core.schemas import InvestmentSchema
from src.utils.logger import setup_logger

logger = setup_logger("core.orchestrator.simulation")

class SimulationOrchestrator:
    """
    Domain Service for orchestrating investment simulations.
    Handles confidence scoring, status classification, and schema enforcement.
    """
    
    def run_simulation(self, forecast: dict, entry_price: float, target_pct: float, investment_amount: float) -> InvestmentSchema:
        """
        Executes the simulation logic and returns a validated InvestmentSchema.
        """
        logger.info(f"Orchestrating simulation for ${investment_amount} at {target_pct}% target")
        
        target_price = entry_price * (1 + target_pct / 100)
        
        # 1. Classification & Exit Projection
        crossing_date = find_target_crossing_date(
            forecast['dates'], 
            forecast['prices'], 
            target_price
        )
        
        status = "SUCCESS" if crossing_date else "TARGET_NOT_REACHED"
        
        # 2. Confidence Scoring (Industrial Logic)
        confidence = self.calculate_confidence_score(forecast, crossing_date)
        
        # 3. Schema Construction
        # Ensure dates are strings for serialization
        forecast_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in forecast['dates']]
        projected_date = crossing_date.strftime('%Y-%m-%d') if hasattr(crossing_date, 'strftime') else str(crossing_date) if crossing_date else None
        
        # Support both ndarray and list
        prices = forecast['prices'].tolist() if hasattr(forecast['prices'], 'tolist') else list(forecast['prices'])
        std = forecast['std'].tolist() if hasattr(forecast['std'], 'tolist') else list(forecast['std'])

        result = InvestmentSchema(
            amount=investment_amount,
            entry_price=entry_price,
            target_pct=target_pct,
            target_price=target_price,
            simulation_status=status,
            confidence_score=confidence,
            projected_withdrawal_date=projected_date,
            forecast_prices=prices,
            forecast_dates=forecast_dates,
            std=std
        )
        
        logger.info(f"Simulation completed with status: {status} (Confidence: {confidence:.2%})")
        return result

    def calculate_confidence_score(self, forecast: dict, crossing_date=None) -> float:
        """
        Calculates a confidence metric based on model standard deviation.
        A score of 1.0 indicates zero uncertainty; lower scores indicate higher volatility.
        """
        try:
            prices = np.array(forecast['prices'])
            stds = np.array(forecast['std'])
            
            if crossing_date:
                # Calculate confidence at the specific moment of exit
                idx = 0
                for i, d in enumerate(forecast['dates']):
                    if d == crossing_date:
                        idx = i
                        break
                
                # Formula: 1 - (2 * StdDev / Mean) -> Representative of the 95% CI relative width
                rel_error = (2 * stds[idx]) / prices[idx]
                score = max(0, 1 - rel_error)
            else:
                # Average confidence across the entire forecast window
                rel_errors = (2 * stds) / prices
                score = max(0, 1 - np.mean(rel_errors))
                
            return float(score)
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.5 # Default conservative value
