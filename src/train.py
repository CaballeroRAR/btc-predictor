import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import cloud_config as cloud_config
from data_loader import prepare_merged_dataset, create_sequences
from model import build_lstm_model

def train_pipeline():
    # 1. Load Core Market Data
    _, df = prepare_merged_dataset()
    
    # 2. Ingest Closed-Loop Enrichment (SYSTEM Drift Logs)
    from database import DatabaseManager
    db = DatabaseManager()
    drift_logs = db.get_live_drift_history(days=30)
    
    if drift_logs:
        print(f"Enriching model with {len(drift_logs)} intraday system snapshots...")
        drift_df = pd.DataFrame(drift_logs)
        drift_df['date'] = pd.to_datetime(drift_df['forecast_date']).dt.date
        
        # Aggregate hourly drift into daily indicators
        daily_drift = drift_df.groupby('date').agg(
            Drift_Alignment=('drift_pct', 'mean'),
            Drift_Volatility=('drift_pct', 'std')
        ).fillna(0)
        
        # Align indexes and merge
        df_dates = df.index.date
        df.index = df_dates
        df = df.join(daily_drift, how='left').fillna(0)
        
        # Restore DatetimeIndex for subsequent processing
        df.index = pd.to_datetime(df.index)
    else:
        # Fallback for cold-start: initialize features with 0
        df['Drift_Alignment'] = 0.0
        df['Drift_Volatility'] = 0.0
    
    # 3. Scale Features
    print(f"Final training matrix shape: {df.shape} ({list(df.columns)})")
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    # Save scaler (Crucial for inference/drift analysis)
    os.makedirs(cloud_config.MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, cloud_config.SCALER_PATH)
    print(f"Saved scaler to {cloud_config.SCALER_PATH}")
    
    # 3. Create Sequences
    X, y = create_sequences(scaled_data)
    
    # Split: 80% Train, 20% validation
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # 4. Build Model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # 5. Train
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(cloud_config.MODEL_PATH, save_best_only=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    print(f"Training complete. Model saved to {cloud_config.MODEL_PATH}")
    
    # Upload to GCS if in cloud environment
    if os.environ.get("GCP_PROJECT"):
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(cloud_config.BUCKET_NAME)
        
        # Upload Model
        model_blob = bucket.blob(f"{cloud_config.MODEL_DIR}/btc_lstm_model.h5")
        model_blob.upload_from_filename(cloud_config.MODEL_PATH)
        
        # Upload Scaler
        scaler_blob = bucket.blob(f"{cloud_config.MODEL_DIR}/scaler.pkl")
        scaler_blob.upload_from_filename(cloud_config.SCALER_PATH)
        
        print("Uploaded model and scaler to GCS.")

if __name__ == "__main__":
    train_pipeline()
