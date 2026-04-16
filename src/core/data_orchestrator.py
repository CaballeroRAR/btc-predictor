import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
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

        # P-06: F&G coverage warning
        if not sentiment_df.empty:
            expected_days = int(cloud_config.YEARS_HISTORY * 365)
            actual_days = len(sentiment_df)
            if actual_days < expected_days:
                self.logger.warning(
                    f"F&G INDEX COVERAGE GAP: API returned {actual_days} days, "
                    f"YEARS_HISTORY={cloud_config.YEARS_HISTORY} requires {expected_days}. "
                    f"Approximately {expected_days - actual_days} days will be forward-filled "
                    f"from the earliest available value."
                )

        wiki_df = self.adapter.fetch_wikipedia_views()
        rss_sentiment = self.adapter.fetch_rss_sentiment()
        
        # 3. Hybrid Signal Processing (Curiosity Multiplier)
        if not wiki_df.empty:
            # P-03: Refinement. Isolate live RSS sentiment to the final observation.
            # This ensures that historical training rows remain "pure" while 
            # the live pulse captures intraday volatility.
            latest_date = wiki_df.index[-1]
            multiplier = 1 + np.clip(rss_sentiment, -0.5, 0.5) # Guard against extreme sentiment spikes
            wiki_df.loc[latest_date, 'Google_Trends'] *= multiplier
            wiki_df['Google_Trends'] = wiki_df['Google_Trends'].clip(0, 100)
            self.logger.info(f"SIGNAL: Applied {rss_sentiment:+.2f} live sentiment multiplier to {latest_date.date()}")
        else:
            self.logger.warning("CORE: Wikipedia views unavailable. Using neutral baseline (50.0).")
            wiki_df = pd.DataFrame({"Google_Trends": [50.0]})

        # 1. Gap Recovery: Harden the yesterday stitch logic
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
        """
        Recovers a missing yesterday close via high-resolution hourly data.
        Refinement: Timezone-aware normalization and robust download fallback.
        """
        now = datetime.now(timezone.utc)
        yesterday_utc = (now - timedelta(days=1)).date()
        
        if not price_df.empty and price_df.index[-1].date() < yesterday_utc:
            self.logger.info(f"GAP DETECTED: Recovering finalized data for {yesterday_utc} UTC...")
            try:
                # Use download for better multi-asset stability in the stitch path
                hist_h = yf.download("BTC-USD", period="2d", interval="1h", progress=False)
                if not hist_h.empty:
                    # Normalize both to UTC for alignment
                    hist_h.index = hist_h.index.tz_convert(None)
                    yesterday_dt = pd.to_datetime(yesterday_utc)
                    
                    # Find the last hourly close of yesterday
                    mask = hist_h.index.normalize() == yesterday_dt
                    if mask.any():
                        y_close = hist_h[mask]['Close'].iloc[-1]
                        y_row = price_df.iloc[-1:].copy()
                        y_row.index = [yesterday_dt]
                        y_row['Close'] = float(y_close)
                        price_df = pd.concat([price_df, y_row])
                        self.logger.info(f"STITCH SUCCESS: Added {yesterday_utc} @ ${float(y_close):,.2f}")
                    else:
                        self.logger.warning(f"STITCH FAILED: No data points found for {yesterday_utc} in 48h window.")
                else:
                    self.logger.warning(f"STITCH FAILED: yfinance returned empty hourly dataset.")
            except Exception as e:
                self.logger.warning(f"STITCH FAILED: Could not recover {yesterday_utc} ({e})")
        return price_df

    def _apply_temporal_guard(self, df):
        """
        Drops incomplete current-day bars if it is too early in the UTC day.
        Ensures the model doesn't train on 'partial' candles.
        """
        now = datetime.now(timezone.utc)
        today = now.date()
        current_hour = now.hour
        
        # Only drop if the last row is Today and it's before 10:00 UTC (Incomplete volume)
        if not df.empty and df.index[-1].date() == today and current_hour < 10:
            self.logger.info(f"GUARD: Dropping early ({current_hour}:00 UTC) today candle.")
            return df.iloc[:-1]
        return df

# Accessor
data_orchestrator = DataOrchestrator()
