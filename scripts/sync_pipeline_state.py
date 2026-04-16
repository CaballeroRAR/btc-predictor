import os
import json
import pandas as pd
from datetime import datetime, timezone
import sys

# Add project root to path for cloud_config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.cloud_config as cloud_config

HISTORY_FILE = os.path.join(".agent", "memory", "pipeline_history.json")

def get_file_metadata(path):
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    return None

def sync():
    if not os.path.exists(HISTORY_FILE):
        history = []
    else:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)

    changed = False

    # 1. Audit ETL
    data_path = os.path.join(cloud_config.DATA_DIR, "merged_data.csv")
    etl_ts = get_file_metadata(data_path)
    if etl_ts:
        # Check if already in history
        if not any(entry["timestamp"] == etl_ts and entry["type"] == "ETL" for entry in history):
            try:
                df = pd.read_csv(data_path)
                count = len(df)
                last_date = df.iloc[-1, 0] if not df.empty else "unknown"
            except Exception:
                count = "error"
                last_date = "error"

            history.append({
                "timestamp": etl_ts,
                "type": "ETL",
                "status": "success",
                "record_count": count,
                "last_index": str(last_date),
                "source": "IndustrialMarketAdapter",
                "environment": "local"
            })
            changed = True

    # 2. Audit Training
    model_ts = get_file_metadata(cloud_config.MODEL_PATH)
    if model_ts:
        if not any(entry["timestamp"] == model_ts and entry["type"] == "TRAINING" for entry in history):
            history.append({
                "timestamp": model_ts,
                "type": "TRAINING",
                "status": "success",
                "metrics": {
                    "model_path": cloud_config.MODEL_PATH,
                    "scaler_path": cloud_config.SCALER_PATH
                },
                "environment": "local"
            })
            changed = True

    if changed:
        # Sort by timestamp descending
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Sync complete. Added {int(changed)} new record(s) to history.")
    else:
        print("Sync complete. No new pipeline state detected.")

if __name__ == "__main__":
    sync()
