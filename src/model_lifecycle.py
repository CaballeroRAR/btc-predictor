import os
import json
from datetime import datetime
import cloud_config as cloud_config

CALIBRATION_FILE = "data/calibration_state.json"

def get_model_info():
    """Returns model age in days and training timestamp."""
    if not os.path.exists(cloud_config.MODEL_PATH):
        return None, None
    
    # Get creation time (platform-dependent but usually reliable for local dev)
    ctime = os.path.getctime(cloud_config.MODEL_PATH)
    dt_trained = datetime.fromtimestamp(ctime)
    age_days = (datetime.now() - dt_trained).days
    
    return age_days, dt_trained

def save_calibration_state(drift_val, price):
    """Saves the calculated drift and the price it was based on."""
    state = {
        "last_calibration_date": str(datetime.now()),
        "drift_value": float(drift_val),
        "reference_price": float(price),
        "model_path": cloud_config.MODEL_PATH
    }
    
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(state, f, indent=4)
    return state

def load_calibration_state():
    """Loads the stored drift. Defaults to 0.0 if not found or if Day 0."""
    age_days, _ = get_model_info()
    
    # If Day 0 (Fresh training), we ignore old drift
    if age_days == 0:
        return {"drift_value": 0.0, "status": "Day 0 (Fresh State)"}
        
    if not os.path.exists(CALIBRATION_FILE):
        return {"drift_value": 0.0, "status": "No Calibration Found"}
        
    try:
        with open(CALIBRATION_FILE, 'r') as f:
            state = json.load(f)
            state["status"] = "Active"
            return state
    except:
        return {"drift_value": 0.0, "status": "Error Loading State"}
