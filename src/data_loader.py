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
import time
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_btc_data(years=cloud_config.YEARS_HISTORY, start_date=None):
    """Fetch historical BTC, ETH, Gold, and Oil data for ratio analysis."""
    end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=years * 365)
    
    print(f"Fetching price data from {start_date.date()} to {end_date.date()}...")
    btc = yf.download("BTC-USD", start=start_date, end=end_date, interval="1d", progress=False)
    eth = yf.download("ETH-USD", start=start_date, end=end_date, interval="1d", progress=False)
    gold = yf.download("GC=F", start=start_date, end=end_date, interval="1d", progress=False)
    dxy = yf.download("DX-Y.NYB", start=start_date, end=end_date, interval="1d", progress=False)
    us10y = yf.download("^TNX", start=start_date, end=end_date, interval="1d", progress=False)
    
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

def fetch_rss_sentiment():
    """Fetch recent news sentiment from RSS feeds using VADER NLP."""
    print("Fetching News Sentiment from RSS feeds...")
    analyzer = SentimentIntensityAnalyzer()
    all_scores = []
    
    for url in cloud_config.RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            # Take the latest 10 headlines from each feed
            entries = feed.entries[:10]
            for entry in entries:
                # VADER compound score is -1 (bearish) to 1 (bullish)
                score = analyzer.polarity_scores(entry.title)['compound']
                all_scores.append(score)
        except Exception as e:
            print(f"Error parsing RSS feed {url}: {e}")
            
    if not all_scores:
        return 50.0 # Neutral fallback
    
    # Map average [-1, 1] to [0, 100] scale
    mean_sentiment = sum(all_scores) / len(all_scores)
    final_score = (mean_sentiment + 1) * 50
    
    print(f"Aggregated RSS Sentiment: {final_score:.2f}")
    return final_score

def prepare_merged_dataset(force_refresh=False):
    """Merge price, sentiment, and trends data with incremental updates."""
    local_path = os.path.join(cloud_config.DATA_DIR, "merged_data.csv")
    os.makedirs(cloud_config.DATA_DIR, exist_ok=True)
    
    existing_df = pd.DataFrame()
    last_date = None
    
    if not force_refresh and os.path.exists(local_path):
        try:
            existing_df = pd.read_csv(local_path, index_col='Date', parse_dates=True)
            if not existing_df.empty:
                last_date = existing_df.index.max()
                print(f"Found existing data until {last_date.date()}.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}")

    # Determine startup range
    if last_date is None:
        years = cloud_config.YEARS_HISTORY
        start_date = datetime.now() - timedelta(days=years * 365)
    else:
        # Fetch from the day after the last entry
        start_date = last_date + timedelta(days=1)
        
    # (If last_date is yesterday/today, we skip yfinance/sentiment fetch)
    if last_date is not None and start_date.date() >= datetime.now().date():
        print("Data is already up to date.")
        clean_df = existing_df.copy()
        if len(clean_df) > 1:
            clean_df = clean_df.iloc[:-1]
        return existing_df, clean_df

    print(f"Fetching incremental data since {start_date.date()}...")
    
    # 1. Fetch Price Delta
    new_price_df = fetch_btc_data(start_date=start_date)
    
    # 2. Fetch Sentiment Delta
    all_sentiment_df = fetch_sentiment_data()
    new_sentiment_df = all_sentiment_df[all_sentiment_df.index >= start_date]
    
    # 3. News Sentiment (Replaces Google Trends baseline to maintain 12-feature schema)
    news_val = fetch_rss_sentiment()
    new_merged_df['News_Sentiment'] = news_val
    
    # Combine with existing
    if not existing_df.empty:
        full_df = pd.concat([existing_df, new_merged_df])
        full_df = full_df[~full_df.index.duplicated(keep='last')].sort_index()
    else:
        full_df = new_merged_df
    
    # Ensure column order matches training: [OHLCV, Ratios, Macro, RSI, Sentiment, News]
    # (Replacing Google_Trends at index 11)
    
    full_df.ffill(inplace=True)
    full_df.dropna(inplace=True)
    
    # Save the updated dataset locally (Full version)
    full_df.to_csv(local_path)
    
    # Create a Cleaned Version for Model Training/Inference (Drop partial today)
    clean_df = full_df.copy()
    if len(clean_df) > 1:
        clean_df = clean_df.iloc[:-1]
    
    print(f"Dataset updated. Full: {full_df.shape}, Clean: {clean_df.shape}")
    
    return full_df, clean_df

# Removed save_to_gcs (Deadcode - Deployment script handles sync via 'gcloud storage cp')

def create_sequences(scaled_data, lookback=cloud_config.LOOKBACK_DAYS, forecast=cloud_config.FORECAST_DAYS):
    """Create multi-step sequences for LSTM."""
    X, y = [], []
    for i in range(len(scaled_data) - lookback - forecast + 1):
        X.append(scaled_data[i : i + lookback])
        y.append(scaled_data[i + lookback : i + lookback + forecast, 3])
    return np.array(X), np.array(y)

if __name__ == "__main__":
    full_df, clean_df = prepare_merged_dataset()
    print(full_df.tail())
