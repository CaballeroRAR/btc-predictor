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
    
    # 2. Reverting to Stabilized 12-Feature Architecture
    # We are removing the experimental 'Closed-Loop' drift enrichment to restore
    # baseline stability until the unit normalization is fully validated.
    print(f"Dataset loaded with {df.shape[1]} features. Enforcing 12 core signals...")
    
    expected_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'BTC_ETH_Ratio', 'BTC_Gold_Ratio', 'DXY', 'US10Y', 'RSI', 
        'Sentiment', 'News_Sentiment'
    ]
    df = df[expected_cols]
    
    # 3. Scale Features
    print(f"Final training matrix shape: {df.shape}")
    print(f"Schema Verification: {list(df.columns)}")
    print("\n--- Feature Drift Signal Distribution ---")
    print(df[['Drift_Alignment', 'Drift_Volatility']].describe().iloc[1:3]) # Mean and Std
    print("------------------------------------------\n")
    
    if df.empty:
        raise ValueError("FATAL: Training dataframe is empty. Cannot proceed with fit_transform.")

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
    
    # 5. Cloud Integration: Check for Vertex AI or GCP Environment
    is_cloud = os.environ.get("AIP_MODEL_DIR") or os.environ.get("GCP_PROJECT") or os.environ.get("PROJECT_ID")
    
    if is_cloud:
        print(f"Cloud environment detected (AIP_MODEL_DIR={os.environ.get('AIP_MODEL_DIR')}). Starting GCS upload...")
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
