import json
import os
import pandas as pd
from datetime import datetime

INVESTMENTS_FILE = "data/investments.json"

def load_investments():
    """
    Load the investment journal from a local JSON file.
    
    Returns:
        list: A list of investment dictionaries. Returns an empty list if file not found.
    """
    if not os.path.exists(INVESTMENTS_FILE):
        return []
    try:
        with open(INVESTMENTS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def save_investment(amount, date, price, forecast_prices=None, calibrated_prices=None, std=None, forecast_dates=None, profit_target=2.0, original_withdrawal_date=None, note=""):
    """
    Append a new investment entry to the journal with a forecast snapshot.
    
    Args:
        amount (float): Investment amount in USD.
        date (str/datetime): Date of purchase.
        price (float): BTC price at time of purchase.
        forecast_prices (list/np.array): Raw model output array.
        calibrated_prices (list/np.array): Drift-aligned model output array.
        std (list/np.array): Standard deviation (uncertainty) bands.
        forecast_dates (list): Dates corresponding to the forecast window.
        profit_target (float): Multiplier for target withdrawal (e.g., 2.0 for 2x).
        original_withdrawal_date (str/datetime): Initially projected exit date.
        note (str): Optional user annotation.
        
    Returns:
        dict: The newly created investment record.
    """
    investments = load_investments()
    
    # Convert numpy arrays to lists for JSON serialization
    fp = forecast_prices.tolist() if hasattr(forecast_prices, "tolist") else (forecast_prices or [])
    cp = calibrated_prices.tolist() if hasattr(calibrated_prices, "tolist") else (calibrated_prices or [])
    sd = std.tolist() if hasattr(std, "tolist") else (std or [])
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
        "original_withdrawal_date": str(original_withdrawal_date) if original_withdrawal_date else None,
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
