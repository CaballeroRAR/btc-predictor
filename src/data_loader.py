import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
import numpy as np
from pytrends.request import TrendReq
from google.cloud import storage
import io
import cloud_config as cloud_config

def fetch_btc_data(years=cloud_config.YEARS_HISTORY):
    """Fetch historical BTC, ETH, Gold, and Oil data for ratio analysis."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    print(f"Fetching BTC, ETH, Gold, DXY, and US10Y price data since {start_date.date()}...")
    btc = yf.download("BTC-USD", start=start_date, interval="1d")
    eth = yf.download("ETH-USD", start=start_date, interval="1d")
    gold = yf.download("GC=F", start=start_date, interval="1d")
    dxy = yf.download("DX-Y.NYB", start=start_date, interval="1d")
    us10y = yf.download("^TNX", start=start_date, interval="1d")
    
    # Handle yfinance MultiIndex columns if present
    def extract_level(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
        
    btc = extract_level(btc)
    eth = extract_level(eth)
    gold = extract_level(gold)
    dxy = extract_level(dxy)
    us10y = extract_level(us10y)
        
    df = btc[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Macro assets skip weekends. We align to BTC's 24/7 calendar and ffill Friday's close.
    gold_close = gold['Close'].reindex(btc.index).ffill()
    dxy_close = dxy['Close'].reindex(btc.index).ffill()
    us10y_close = us10y['Close'].reindex(btc.index).ffill()
    
    # Separate Risk Proxies
    df['BTC_ETH_Ratio'] = btc['Close'] / eth['Close']
    df['BTC_Gold_Ratio'] = btc['Close'] / gold_close
    
    # Macro Gravity Features
    df['DXY'] = dxy_close
    df['US10Y'] = us10y_close
    
    # Technical Sentiment: RSI (14-day)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def fetch_google_trends(keyword="Bitcoin", years=cloud_config.YEARS_HISTORY):
    """Fetch Google Trends interest over time."""
    print(f"Fetching Google Trends for '{keyword}'...")
    pytrends = TrendReq(hl='en-US', tz=360)
    
    try:
        pytrends.build_payload([keyword], cat=0, timeframe='today 5-y', gprop='') # 5-y is max for daily-ish
        df = pytrends.interest_over_time()
        
        if not df.empty:
            df = df[[keyword]].rename(columns={keyword: 'Google_Trends'})
            return df
    except Exception as e:
        print(f"Failed to fetch Google Trends (Rate Limit): {e}")
        
    return pd.DataFrame()

def fetch_sentiment_data():
    """Fetch historical Crypto Fear & Greed Index data."""
    print("Fetching Crypto Fear & Greed Index data...")
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
        df['value'] = df['value'].astype(int)
        df = df[['timestamp', 'value']].rename(columns={'timestamp': 'Date', 'value': 'Sentiment'})
        df.set_index('Date', inplace=True)
        return df
    else:
        raise Exception("Failed to fetch sentiment data")

def prepare_merged_dataset():
    """Merge price, sentiment, and trends data."""
    price_df = fetch_btc_data()
    sentiment_df = fetch_sentiment_data()
    trends_df = fetch_google_trends()
    
    # Align indexes (Price data is the master timeline)
    merged_df = price_df.join(sentiment_df, how='left')
    
    if not trends_df.empty:
        # Trends usually need resampling/reindexing as they might be weekly
        merged_df = merged_df.join(trends_df, how='left')
    else:
        # Maintain 12-feature dimension when Google Trends is rate-limited
        merged_df['Google_Trends'] = 50.0
    
    # Forward fill missing sentiment/trends (as they update less frequently)
    merged_df.ffill(inplace=True)
    merged_df.dropna(inplace=True)
    
    # Drop the live (incomplete) today candle to avoid prediction gap anomalies
    merged_df = merged_df.iloc[:-1]
    
    print(f"Merged dataset shape: {merged_df.shape}")
    return merged_df

def save_to_gcs(df, filename):
    """Upload dataframe to GCS bucket as CSV."""
    client = storage.Client(project=cloud_config.PROJECT_ID)
    bucket = client.bucket(cloud_config.BUCKET_NAME)
    blob = bucket.blob(f"{cloud_config.DATA_DIR}/{filename}")
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
    print(f"Uploaded {filename} to gs://{cloud_config.BUCKET_NAME}")

def create_sequences(scaled_data, lookback=cloud_config.LOOKBACK_DAYS, forecast=cloud_config.FORECAST_DAYS):
    """
    Create multi-step sequences for LSTM (Original 9-feature Raw Price logic).
    """
    X, y = [], []
    for i in range(len(scaled_data) - lookback - forecast + 1):
        # Input features: Scaled OHLCV + Sentiment + Trends + Ratios
        X.append(scaled_data[i : i + lookback])
        
        # Output: Next forecast days closing price (Target at index 3: Close)
        y.append(scaled_data[i + lookback : i + lookback + forecast, 3])
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    df = prepare_merged_dataset()
    print(df.tail())
    # Ensure local directory exists
    os.makedirs(cloud_config.DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(cloud_config.DATA_DIR, "merged_data.csv"))
    
    # Uncomment when GCP credentials are active
    # save_to_gcs(df, "merged_data.csv")
