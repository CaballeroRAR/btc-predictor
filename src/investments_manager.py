import json
import os
import pandas as pd
from datetime import datetime

INVESTMENTS_FILE = "data/investments.json"

def load_investments(db_mgr=None):
    """Load investments from Firestore."""
    from database import DatabaseManager
    if db_mgr is None:
        db_mgr = DatabaseManager()
    return db_mgr.get_investments()

def save_investment(amount, date, price, forecast_prices=None, calibrated_prices=None, std=None, forecast_dates=None, profit_target=2.0, original_withdrawal_date=None, note="", db_mgr=None):
    """Append a new investment to the Firestore journal with forecast snapshot."""
    from database import DatabaseManager
    if db_mgr is None:
        db_mgr = DatabaseManager()
        
    # Convert numpy arrays to lists for JSON serialization
    fp = forecast_prices.tolist() if forecast_prices is not None else []
    cp = calibrated_prices.tolist() if calibrated_prices is not None else []
    sd = std.tolist() if std is not None else []
    fd = [str(d) for d in forecast_dates] if forecast_dates is not None else []
    
    new_inv = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "amount": amount,
        "date": str(date),
        "price": price,
        "profit_target": profit_target,
        "original_withdrawal_date": str(original_withdrawal_date) if original_withdrawal_date else None,
        "forecast_prices": fp,
        "calibrated_prices": cp,
        "std": sd,
        "forecast_dates": fd,
        "note": note,
        "timestamp": str(datetime.now())
    }
    
    db_mgr.save_investment(new_inv)
    return new_inv

def delete_investment(inv_id, db_mgr=None):
    """Remove an investment by ID."""
    from database import DatabaseManager
    if db_mgr is None:
        db_mgr = DatabaseManager()
    db_mgr.delete_investment(inv_id)
