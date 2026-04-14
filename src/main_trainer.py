import os
import sys
# Path resolution for industrial architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
from src import cloud_config
from src.data_loader import prepare_merged_dataset, create_sequences
from src.model import build_lstm_model
from src.utils.logger import setup_logger
from src.repositories.asset_repo import AssetRepository
from src.facades.lifecycle_facade import LifecycleFacade

logger = setup_logger("trainer.main")
assets = AssetRepository()
lifecycle_manager = LifecycleFacade()

def train_pipeline():
    # 1. Load Data
    _, df = prepare_merged_dataset()
    
    # 2. Schema Selection (Stationary Log-Return Architecture)
    from src.core.standardizer import MarketStandardizer
    df = df[MarketStandardizer.REQUIRED_COLUMNS]
    logger.info(f"Training on {df.shape[1]} features. Schema: Stationary Log-Return (2.0)")

    # 3. Scale Features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)
    
    # Save scaler (Crucial for inference/drift analysis)
    assets.save_scaler(scaler, "scaler.pkl")
    
    # 3b. Create Sequences
    X, y = create_sequences(scaled_data)
    
    # Split: 80% Train, 20% validation
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    logger.info(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}")
    
    # 4. Build Model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # 5. Train (100-Cycle High Precision)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(cloud_config.MODEL_PATH, save_best_only=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    logger.info(f"Training complete. Weights saved to {cloud_config.MODEL_PATH}")
    
    # Detect Cloud Environment
    is_cloud = os.environ.get("GCP_PROJECT") or os.environ.get("AIP_MODEL_DIR")
    
    if is_cloud:
        logger.info(f"Initializing Cloud Persistence for gs://{cloud_config.BUCKET_NAME}")
        # Use our new publishing logic to push the fresh model/scaler to GCP
        success = lifecycle_manager.publish_assets()
        if success:
            logger.info("SUCCESS: Training artifacts published to GCS.")
        else:
            logger.error("FAILURE: Training artifacts could not be published.")
    else:
        logger.info("Local environment detected. Skipping GCS sync.")

if __name__ == "__main__":
    train_pipeline()
