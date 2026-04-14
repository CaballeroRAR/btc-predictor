import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.logger import setup_logger
from src.adapters.market_adapter import IndustrialMarketAdapter
from src.core.standardizer import MarketStandardizer
import src.cloud_config as cloud_config

class DataOrchestrator:
    """
    Domain Service for orchestrating market data ingestion and normalization.
    Handles caching, gap-filling, and cross-source alignment.
    """
    def __init__(self):
        self.logger = setup_logger("core.orchestrator")
        self.adapter = IndustrialMarketAdapter()
        self.standardizer = MarketStandardizer()

    def prepare_dataset(self, force_refresh=False):
        """Orchestrates the 20-feature Stationary Log-Return dataset."""
        data_path = os.path.join(cloud_config.DATA_DIR, "merged_data.csv")
        
        # 1. Cache Layer (Check for schema parity)
        if not force_refresh and os.path.exists(data_path):
            self.logger.info("CORE: Delivering cached dataset (High Speed Path)")
            cache_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if cache_df.shape[1] == len(MarketStandardizer.REQUIRED_COLUMNS):
                return cache_df
                
        # 2. Ingestion Phase
        self.logger.info("CORE: Initiating multi-source market ingestion...")
        price_df = self.adapter.fetch_price_data()
        sentiment_df = self.adapter.fetch_fng_sentiment()
        wiki_df = self.adapter.fetch_wikipedia_views()
        rss_sentiment = self.adapter.fetch_rss_sentiment()
        chain_df = self.adapter.fetch_blockchain_metrics()
        
        # 3. Hybrid Signal Processing (Curiosity Multiplier)
        if not wiki_df.empty:
            wiki_df['Google_Trends'] = wiki_df['Google_Trends'] * (1 + rss_sentiment)
            wiki_df['Google_Trends'] = wiki_df['Google_Trends'].clip(0, 100)
        else:
            self.logger.warning("CORE: Wikipedia views unavailable. Using neutral baseline.")
            wiki_df = pd.DataFrame({"Google_Trends": [50.0]})

        # 4. Stationary Transformation (The Log-Return Target)
        # Calculates daily percentage change in log-space (Stationary)
        price_df['Log_Return'] = np.log(price_df['Close'] / price_df['Close'].shift(1))
        
        # 5. Volatility Factor (ATR - 14-period)
        high_low = price_df['High'] - price_df['Low']
        high_close = np.abs(price_df['High'] - price_df['Close'].shift(1))
        low_close = np.abs(price_df['Low'] - price_df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        price_df['ATR'] = true_range.rolling(window=14).mean()
        
        # 6. Temporal Context (Cyclical Fourier Embeddings)
        # Teaches the model day-of-week and month-of-year seasonalities
        days_in_week = 7
        months_in_year = 12
        price_df['Day_Sin'] = np.sin(2 * np.pi * price_df.index.dayofweek / days_in_week)
        price_df['Day_Cos'] = np.cos(2 * np.pi * price_df.index.dayofweek / days_in_week)
        price_df['Month_Sin'] = np.sin(2 * np.pi * price_df.index.month / months_in_year)
        price_df['Month_Cos'] = np.cos(2 * np.pi * price_df.index.month / months_in_year)

        # 7. Alignment Phase
        merged_df = price_df.join(sentiment_df, how='left')
        merged_df = merged_df.join(wiki_df, how='left')
        
        if not chain_df.empty:
            merged_df = merged_df.join(chain_df, how='left')
        else:
            self.logger.warning("CORE: Chain metrics unavailable. Using zero-fill.")
            merged_df['Hashrate'] = 0.0
            merged_df['Difficulty'] = 0.0
        
        merged_df.ffill(inplace=True)
        merged_df.dropna(inplace=True)
        
        # Guard: Truncate early/incomplete today candle
        merged_df = self._apply_temporal_guard(merged_df)
        
        # Enforce Schema (20 features)
        final_df = self.standardizer.enforce_schema(merged_df)
        
        # Persist
        os.makedirs(cloud_config.DATA_DIR, exist_ok=True)
        final_df.to_csv(data_path)
        
        return final_df

    def _apply_temporal_guard(self, df):
        """Drops incomplete current-day bars if it is too early in the day."""
        today = datetime.now().date()
        current_hour = datetime.now().hour
        if not df.empty and df.index[-1].date() == today and current_hour < 10:
            self.logger.info("GUARD: Dropping early (incomplete) today candle.")
            return df.iloc[:-1]
        return df

# Accessor
data_orchestrator = DataOrchestrator()
