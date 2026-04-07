import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add src to sys.path to import prediction_logger
sys.path.append(os.path.join(os.getcwd(), "src"))

from prediction_logger import log_predictions, LOG_FILE, get_performance_stats

def test_logger():
    """Verify that predictions are logged and pruned correctly."""
    # Ensure LOG_FILE does not exist for a fresh test
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    # 1. Test intra-day logging
    today = datetime.now()
    forecast_dates = [today + timedelta(days=i) for i in range(3)]
    
    # Run 2 predictions for the same dates to simulate intra-day heartbeats
    log_predictions(forecast_dates, [70000, 71000, 72000])
    log_predictions(forecast_dates, [70500, 71500, 72500])
    
    df = pd.read_csv(LOG_FILE)
    print(f"Intra-day heartbeat count (Expected 6): {len(df)}")
    if 'sim_run_date' in df.columns:
        print(f"PASS: sim_run_date present: {df['sim_run_date'].iloc[0]}")
    
    # 2. Test Pruning (Simulate old data)
    # Manual insertion of a record older than 7 days
    old_date = datetime.now() - timedelta(days=10)
    old_target = datetime.now() - timedelta(days=10)
    old_data = pd.DataFrame([
        {"timestamp": "Aggregated", "sim_run_date": "2024-01-01", "forecast_date": old_target.strftime("%Y-%m-%d"), "predicted_price": 50000},
        {"timestamp": "Aggregated", "sim_run_date": "2024-01-01", "forecast_date": old_target.strftime("%Y-%m-%d"), "predicted_price": 60000},
    ])
    
    # Append old data manually
    df_combined = pd.concat([pd.read_csv(LOG_FILE), old_data], ignore_index=True)
    df_combined.to_csv(LOG_FILE, index=False)
    
    # Trigger a new log to run pruning
    log_predictions([today + timedelta(days=10)], [80000])
    
    df_pruned = pd.read_csv(LOG_FILE)
    # Check if aggregation happened for 2024-01-01
    agg_subset = df_pruned[df_pruned['sim_run_date'] == "2024-01-01"]
    if len(agg_subset) == 1:
        print("Success: Old entries ( > 7 days) were aggregated by sim_run_date.")
    else:
        print(f"Failure: Pruning logic did not aggregate old data. Length: {len(agg_subset)}")
        print(df_pruned)

if __name__ == "__main__":
    test_logger()
