import json
import os
import pandas as pd
from datetime import datetime

INVESTMENTS_FILE = "data/investments.json"

def load_investments():
    """Load investments from JSON file."""
    if not os.path.exists(INVESTMENTS_FILE):
        return []
    try:
        with open(INVESTMENTS_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_investment(amount, date, price, forecast_prices=None, calibrated_prices=None, std=None, forecast_dates=None, profit_target=2.0, note=""):
    """Append a new investment to the journal with forecast snapshot."""
    investments = load_investments()
    
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
        "forecast_prices": fp,
        "calibrated_prices": cp,
        "std": sd,
        "forecast_dates": fd,
        "note": note,
        "timestamp": str(datetime.now())
    }
    
    investments.append(new_inv)
    os.makedirs(os.path.dirname(INVESTMENTS_FILE), exist_ok=True)
    with open(INVESTMENTS_FILE, 'w') as f:
        json.dump(investments, f, indent=4)
    return new_inv

def delete_investment(inv_id):
    """Remove an investment by ID."""
    investments = load_investments()
    investments = [inv for inv in investments if inv['id'] != inv_id]
    with open(INVESTMENTS_FILE, 'w') as f:
        json.dump(investments, f, indent=4)
