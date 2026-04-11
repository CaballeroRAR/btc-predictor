import pandas as pd
import os
from datetime import datetime
from database import DatabaseManager
import data_loader as data_loader

# Initialize Database Manager
db_manager = DatabaseManager()

def log_predictions(forecast_dates, predicted_prices):
    """
    Append new predictions to the Firestore database.
    sim_run_date: The date this simulation was executed.
    forecast_date: The date being predicted.
    """
    now = datetime.now()
    sim_run_date = now.strftime("%Y-%m-%d")
    
    for d, p in zip(forecast_dates, predicted_prices):
        forecast_date_str = d.strftime("%Y-%m-%d")
        db_manager.log_prediction(
            sim_run_date=sim_run_date,
            forecast_date=forecast_date_str,
            predicted_price=float(p)
        )
    
    print(f"Logged {len(forecast_dates)} predictions to Firestore for run date {sim_run_date}")

def update_actuals(full_df=None):
    """
    Fetch recent actual closing prices and update matching predictions in Firestore.
    Accepts full_df directly to avoid redundant fetching.
    """
    try:
        if full_df is None:
            full_df, _ = data_loader.prepare_merged_dataset()
            
        if full_df.empty:
            return 0
            
        # Get finalized daily closes only (exclude the last unfinalized row)
        recent_actuals = full_df['Close'].iloc[:-1].tail(7)
        
        updated_total = 0
        for date, actual_price in recent_actuals.items():
            date_str = date.strftime("%Y-%m-%d")
            count = db_manager.update_actual_price(date_str, actual_price)
            updated_total += count
            
        print(f"Updated {updated_total} prediction records with actual prices.")
        return updated_total
    except Exception as e:
        print(f"Error updating actuals: {e}")
        return 0

def get_performance_stats():
    """
    Returns a dataframe of historical predictions and actuals for UI plotting.
    """
    history = db_manager.get_prediction_history(limit=200)
    if not history:
        return pd.DataFrame()
        
    df = pd.DataFrame(history)
    # Ensure columns for plotting
    required = ['forecast_date', 'predicted_price', 'actual_price', 'sim_run_date']
    for col in required:
        if col not in df.columns:
            df[col] = None
            
    return df
