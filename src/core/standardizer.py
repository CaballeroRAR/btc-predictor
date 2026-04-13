import numpy as np
import pandas as pd
from src.utils.logger import setup_logger
import src.cloud_config as cloud_config

logger = setup_logger("core.standardizer")

class MarketStandardizer:
    """
    Enforces the 'Macro Gravity' schema (12 features) across all data pipelines.
    Ensures that inferred data matches the training distribution.
    """
    REQUIRED_COLUMNS = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'BTC_ETH_Ratio', 'BTC_Gold_Ratio', 'DXY', 'US10Y', 'RSI', 
        'Sentiment', 'Google_Trends'
    ]

    @classmethod
    def enforce_schema(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Filter and order columns to strictly match the 12-feature tensor requirements."""
        logger.info(f"Enforcing 12-feature schema on dataset of shape {df.shape}")
        
        # Check for missing columns
        missing = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            logger.error(f"SCHEMA BREAK DETECTED. Missing features: {missing}")
            raise ValueError(f"Dataset missing required features: {missing}")
            
        return df[cls.REQUIRED_COLUMNS].copy()

    @classmethod
    def create_sequences(cls, scaled_data, lookback=cloud_config.LOOKBACK_DAYS, forecast=cloud_config.FORECAST_DAYS):
        """
        Create multi-step sequences for LSTM (Original 12-feature logic).
        """
        logger.info(f"Generating sequences: Lookback={lookback}, Forecast={forecast}")
        X, y = [], []
        for i in range(len(scaled_data) - lookback - forecast + 1):
            # Input features: Scaled tensor (12 features)
            X.append(scaled_data[i : i + lookback])
            
            # Output: Next forecast days closing price (Target at index 3: Close)
            y.append(scaled_data[i + lookback : i + lookback + forecast, 3])
            
        return np.array(X), np.array(y)
