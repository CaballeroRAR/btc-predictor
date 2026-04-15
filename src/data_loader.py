import os
import io
import numpy as np
import pandas as pd
from google.cloud import storage
from src import cloud_config
from src.utils.logger import setup_logger
from src.adapters.market_adapter import IndustrialMarketAdapter
from src.core.data_orchestrator import data_orchestrator
from src.core.standardizer import MarketStandardizer

logger = setup_logger("core.loader")
adapter = IndustrialMarketAdapter()

# P-05: Conditional streamlit import. The trainer imports this module and must not
# require streamlit as a container dependency during Vertex AI training jobs.
try:
    import streamlit as st

    @st.cache_data(ttl=3600)
    def get_last_hour_price_with_cache():
        """Industrial Proxy: Fetches hourly price via MarketAdapter. Streamlit-cached."""
        return adapter.fetch_price_data(years=1/365).iloc[-1]['Close']

except ImportError:
    def get_last_hour_price_with_cache():
        """Industrial Proxy: Fetches hourly price via MarketAdapter (no Streamlit cache)."""
        return adapter.fetch_price_data(years=1/365).iloc[-1]['Close']


def fetch_btc_data(years=cloud_config.YEARS_HISTORY):
    """Industrial Proxy: Fetches historical prices via MarketAdapter."""
    return adapter.fetch_price_data(years=years)


def fetch_wikipedia_views(article="Bitcoin", years=cloud_config.YEARS_HISTORY):
    """Industrial Proxy: Fetches Wikipedia pageviews via MarketAdapter."""
    return adapter.fetch_wikipedia_views(article=article, years=years)


def fetch_rss_sentiment():
    """Industrial Proxy: Fetches RSS sentiment via MarketAdapter."""
    return adapter.fetch_rss_sentiment()


def fetch_wikipedia_hourly(article="Bitcoin"):
    """Industrial Proxy: Fetches hourly views via MarketAdapter."""
    return adapter.fetch_hourly_views(article=article)


def fetch_sentiment_data():
    """Industrial Proxy: Fetches Fear and Greed sentiment via MarketAdapter."""
    return adapter.fetch_fng_sentiment()


def prepare_merged_dataset(force_refresh=False):
    """Industrial Proxy: Orchestrates dataset preparation via DataOrchestrator."""
    df = data_orchestrator.prepare_dataset(force_refresh=force_refresh)
    return df, df  # Return tuple for dashboard and trainer compatibility


def save_to_gcs(df, filename):
    """Upload dataframe to GCS bucket as CSV."""
    client = storage.Client(project=cloud_config.PROJECT_ID)
    bucket = client.bucket(cloud_config.BUCKET_NAME)
    blob = bucket.blob(f"{cloud_config.DATA_DIR}/{filename}")

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
    logger.info(f"Uploaded {filename} to gs://{cloud_config.BUCKET_NAME}")


def create_sequences(scaled_data, lookback=cloud_config.LOOKBACK_DAYS, forecast=cloud_config.FORECAST_DAYS):
    """Industrial Proxy: Delegates sequence creation to MarketStandardizer."""
    return MarketStandardizer.create_sequences(scaled_data, lookback, forecast)


if __name__ == "__main__":
    df, _ = prepare_merged_dataset()
    print(df.tail())
    os.makedirs(cloud_config.DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(cloud_config.DATA_DIR, "merged_data.csv"))
