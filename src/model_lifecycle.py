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
def load_assets():
    """Download from GCS if missing, then load model and scaler into memory."""
    os.makedirs(cloud_config.MODEL_DIR, exist_ok=True)

    # 1. Sync from GCS if local is missing (Cloud Run Cold Start)
    if not os.path.exists(cloud_config.MODEL_PATH) or not os.path.exists(cloud_config.SCALER_PATH):
        with st.status("Syncing model from Google Cloud Storage...", expanded=True) as status:
            try:
                client = storage.Client()
                bucket = client.bucket(cloud_config.BUCKET_NAME)
                
                # Download Model
                if not os.path.exists(cloud_config.MODEL_PATH):
                    st.write("Downloading model...")
                    blob = bucket.blob(f"{cloud_config.MODEL_DIR}/btc_lstm_model.h5")
                    blob.download_to_filename(cloud_config.MODEL_PATH)
                
                # Download Scaler
                if not os.path.exists(cloud_config.SCALER_PATH):
                    st.write("Downloading scaler...")
                    blob = bucket.blob(f"{cloud_config.MODEL_DIR}/scaler.pkl")
                    blob.download_to_filename(cloud_config.SCALER_PATH)
                
                status.update(label="Sync Complete!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="Sync Failed!", state="error", expanded=True)
                raise RuntimeError(f"Could not retrieve model from GCS: {str(e)}")
    
    # 2. Final Load
    try:
        model = keras.models.load_model(cloud_config.MODEL_PATH, compile=False)
        scaler = joblib.load(cloud_config.SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None

def get_model_info():
    """Returns model age in days and training timestamp."""
    if not os.path.exists(cloud_config.MODEL_PATH):
        return None, None
    
    # Get creation time
    ctime = os.path.getctime(cloud_config.MODEL_PATH)
    dt_trained = datetime.fromtimestamp(ctime)
    
    # Date-based age (e.g., trained yesterday = 1 day old)
    age_days = (datetime.now().date() - dt_trained.date()).days
    
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
