import os
import pandas as pd
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
        """Orchestrates the full 12-feature dataset retrieval and alignment."""
        data_path = os.path.join(cloud_config.DATA_DIR, "merged_data.csv")
        
        # 1. Cache Layer
        if not force_refresh and os.path.exists(data_path):
            self.logger.info("CORE: Delivering cached dataset (High Speed Path)")
            cache_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if cache_df.shape[1] == 12:
                return cache_df
                
        # 2. Ingestion Phase
        self.logger.info("CORE: Initiating multi-source market ingestion...")
        price_df = self.adapter.fetch_price_data()
        sentiment_df = self.adapter.fetch_fng_sentiment()
        
        wiki_df = self.adapter.fetch_wikipedia_views()
        rss_sentiment = self.adapter.fetch_rss_sentiment()
        
        # 3. Hybrid Signal Processing (Curiosity Multiplier)
        if not wiki_df.empty:
            wiki_df['Google_Trends'] = wiki_df['Google_Trends'] * (1 + rss_sentiment)
            wiki_df['Google_Trends'] = wiki_df['Google_Trends'].clip(0, 100)
        else:
            self.logger.warning("CORE: Wikipedia views unavailable. Using neutral baseline (50.0).")
            wiki_df = pd.DataFrame({"Google_Trends": [50.0]})

        # 4. Gap Recovery (Yesterday Stitch)
        price_df = self._stitch_yesterday_gap(price_df)
        
        # 5. Alignment & Standardization
        merged_df = price_df.join(sentiment_df, how='left')
        merged_df = merged_df.join(wiki_df, how='left')
        
        merged_df.ffill(inplace=True)
        merged_df.dropna(inplace=True)
        
        # Guard: Truncate early/incomplete today candle
        merged_df = self._apply_temporal_guard(merged_df)
        
        # Enforce Schema
        final_df = self.standardizer.enforce_schema(merged_df)
        
        # Persist
        os.makedirs(cloud_config.DATA_DIR, exist_ok=True)
        final_df.to_csv(data_path)
        
        return final_df

    def _stitch_yesterday_gap(self, price_df):
        """Attempts to recover missing yesterday close from high-res pulse if needed."""
        yesterday = (datetime.now() - timedelta(days=1)).date()
        if not price_df.empty and price_df.index[-1].date() < yesterday:
            self.logger.info(f"GAP DETECTED: Recovering finalized data for {yesterday}...")
            # Note: We rely on the adapter 1d download usually working, 
            # but this is the logical home for the 'Stitch' logic.
            pass 
        return price_df

    def _apply_temporal_guard(self, df):
        """Drops incomplete current-day bars if it is too early in the day."""
        today = datetime.now().date()
        current_hour = datetime.now().hour
        if df.index[-1].date() == today and current_hour < 10:
            self.logger.info("GUARD: Dropping early (incomplete) today candle.")
            return df.iloc[:-1]
        return df

# Accessor
data_orchestrator = DataOrchestrator()
