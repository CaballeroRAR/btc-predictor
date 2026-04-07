import pandas as pd
import os
from datetime import datetime, timedelta
import cloud_config as cloud_config

LOG_FILE = "data/prediction_history.csv"

def log_predictions(forecast_dates, predicted_prices):
    """
    Append new predictions to the historical log.
    sim_run_date: The date this simulation was executed.
    forecast_date: The date being predicted.
    """
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    now = datetime.now()
    new_data = []
    for d, p in zip(forecast_dates, predicted_prices):
        new_data.append({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "sim_run_date": now.strftime("%Y-%m-%d"),
            "forecast_date": d.strftime("%Y-%m-%d"),
            "predicted_price": float(p)
        })
    
    df_new = pd.DataFrame(new_data)
    
    if os.path.exists(LOG_FILE):
        df_existing = pd.read_csv(LOG_FILE)
        # Ensure sim_run_date exists in old data (backwards compatibility)
        if 'sim_run_date' not in df_existing.columns:
            df_existing['sim_run_date'] = df_existing['timestamp'].str[:10]
            
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
        
    # Run Pruning / Aggregation
    df_combined = prune_old_predictions(df_combined)
    
    df_combined.to_csv(LOG_FILE, index=False)
    print(f"Logged {len(new_data)} predictions for run date {now.date()}")

def prune_old_predictions(df):
    """
    Compress entries older than 7 days into daily means.
    """
    if df.empty:
        return df
        
    df['forecast_date_dt'] = pd.to_datetime(df['forecast_date'])
    today = datetime.now().date()
    
    # Identify entries older than 7 days
    threshold_date = today - timedelta(days=7)
    
    mask_old = df['forecast_date_dt'].dt.date < threshold_date
    df_old = df[mask_old]
    df_recent = df[~mask_old]
    
    if df_old.empty:
        return df.drop(columns=['forecast_date_dt'])
    
    # Aggregate old entries by forecast_date
    df_old_agg = df_old.groupby(['sim_run_date', 'forecast_date']).agg({
        'timestamp': lambda x: 'Aggregated',
        'predicted_price': 'mean'
    }).reset_index()
    
    # Re-add columns to match recent data
    df_old_agg = df_old_agg[['timestamp', 'sim_run_date', 'forecast_date', 'predicted_price']]
    
    df_final = pd.concat([df_old_agg, df_recent.drop(columns=['forecast_date_dt'])], ignore_index=True)
    return df_final

def get_performance_stats():
    """
    Returns dataframes for dual-showcase metrics.
    """
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    return df
