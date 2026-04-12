import pandas as pd
from src.utils.logger import setup_logger

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
