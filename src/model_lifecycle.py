import os
import json
import keras
import joblib
import streamlit as st
from datetime import datetime
from google.cloud import storage
import cloud_config as cloud_config

CALIBRATION_FILE = "data/calibration_state.json"

@st.cache_resource
def get_active_model():
    """Download from GCS if missing, then load model and scaler into memory. Context-aware."""
    os.makedirs(cloud_config.MODEL_DIR, exist_ok=True)

    # 1. Sync from GCS if local is missing
    if not os.path.exists(cloud_config.MODEL_PATH) or not os.path.exists(cloud_config.SCALER_PATH):
        import sys
        is_streamlit = "streamlit" in sys.modules
        
        if is_streamlit:
            with st.status("Syncing model from Google Cloud Storage...", expanded=True) as status:
                try:
                    force_sync_from_gcs(check_exists=True)
                    status.update(label="Sync Complete!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Sync Failed!", state="error", expanded=True)
                    raise RuntimeError(f"Could not retrieve model from GCS: {str(e)}")
        else:
            print(f"[{datetime.now()}] [SYSTEM] Headless Sync: Downloading assets...")
            force_sync_from_gcs(check_exists=True)

    # 2. Final Load
    try:
        model = keras.models.load_model(cloud_config.MODEL_PATH, compile=False)
        scaler = joblib.load(cloud_config.SCALER_PATH)
        return model, scaler
    except Exception as e:
        if "streamlit" in sys.modules:
            st.error(f"Error loading model files: {str(e)}")
        else:
            print(f"[{datetime.now()}] [ERROR] Model load failure: {str(e)}")
        return None, None

def force_sync_from_gcs(check_exists=False):
    """
    Downloads model and scaler from GCS. 
    If check_exists=True, it only downloads what is missing.
    If check_exists=False, it overwrites local files with cloud versions.
    """
    client = storage.Client()
    bucket = client.bucket(cloud_config.BUCKET_NAME)
    
    # Model File
    if not check_exists or not os.path.exists(cloud_config.MODEL_PATH):
        blob = bucket.blob(f"{cloud_config.MODEL_DIR}/btc_lstm_model.h5")
        blob.download_to_filename(cloud_config.MODEL_PATH)
        m_time = os.path.getmtime(cloud_config.MODEL_PATH)
        print(f"[{datetime.now()}] [GCS] Model downloaded. Local mtime: {m_time}")

    # Scaler File
    if not check_exists or not os.path.exists(cloud_config.SCALER_PATH):
        blob = bucket.blob(f"{cloud_config.MODEL_DIR}/scaler.pkl")
        blob.download_to_filename(cloud_config.SCALER_PATH)
        print(f"[{datetime.now()}] [GCS] Scaler downloaded.")

def get_model_info():
    """Returns model age in days and training timestamp based on last modification."""
    if not os.path.exists(cloud_config.MODEL_PATH):
        return None, None
    
    # Use modification time as it updates when the model is replaced/overwritten
    mtime = os.path.getmtime(cloud_config.MODEL_PATH)
    dt_updated = datetime.fromtimestamp(mtime)
    
    # Date-based age (e.g., updated today = 0 days old)
    age_days = (datetime.now().date() - dt_updated.date()).days
    
    return age_days, dt_updated

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

# --- Aliases for Backward Compatibility ---

