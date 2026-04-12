import numpy as np
import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger("core.simulation")

def calculate_roi(entry_price: float, current_price: float):
    """Calculate Return on Investment percentage."""
    return ((current_price / entry_price) - 1) * 100

def find_target_crossing_date(dates, prices, target_price: float):
    """
    Finds the first date in a forecast vector where the price exceeds a target.
    Returns None if the target is never reached.
    """
    for d, p in zip(dates, prices):
        if p >= target_price:
            logger.info(f"Target ${target_price:,.0f} crossing detected on {d}")
            return d
    return None

def calculate_withdrawal_date(dates, prices, target_price: float):
    """
    Alias for find_target_crossing_date to maintain naming parity with legacy logic.
    """
    return find_target_crossing_date(dates, prices, target_price)

def compute_withdrawal_plan(base_forecast: dict, entry_price: float, profit_target_pct: float, investment_amt: float):
    """
    Pure business logic for investment simulation.
    Takes a forecast result and projects entry/exit benchmarks.
    """
    logger.info("Computing withdrawal plan for new simulation parameters")
    
    target_price = entry_price * (1 + profit_target_pct / 100)
    crossing_date = find_target_crossing_date(base_forecast['dates'], base_forecast['prices'], target_price)
    
    return {
        **base_forecast,
        'target_price': target_price,
        'target_pct': profit_target_pct,
        'entry_price': entry_price,
        'investment_amount': investment_amt,
        'projected_withdrawal_date': crossing_date
    }
