import pandas as pd
import yfinance as yf
import requests
import json
import os
from datetime import datetime, timedelta
import numpy as np
from google.cloud import storage
import io
import cloud_config as cloud_config
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
        
    # (If last_date is yesterday/today, we skip yfinance/sentiment fetch UNLESS force_refresh is True)
    if not force_refresh and last_date is not None and start_date.date() >= datetime.now().date():
        print("Data is already up to date. (Skipping network fetch)")
        clean_df = existing_df.copy()
        if len(clean_df) > 1:
            clean_df = clean_df.iloc[:-1]
        return existing_df, clean_df

    print(f"Fetching incremental data since {start_date.date()}...")
    
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 1. Start parallel fetch tasks
        price_task = executor.submit(fetch_btc_data, start_date=start_date)
        sentiment_task = executor.submit(fetch_sentiment_data)
        news_task = executor.submit(fetch_rss_sentiment)
        
        # 2. Gather results
        new_price_df = price_task.result()
        all_sentiment_df = sentiment_task.result()
        news_val = news_task.result()

    # 3. Filter Sentiment Delta
    new_sentiment_df = all_sentiment_df[all_sentiment_df.index >= start_date]
    
    # 4. Join Price and Sentiment Delta
    new_merged_df = new_price_df.join(new_sentiment_df, how='left')
    
    # 5. News Sentiment (already fetched in parallel)
    new_merged_df['News_Sentiment'] = news_val
    
    # Combine with existing
    if not existing_df.empty:
        full_df = pd.concat([existing_df, new_merged_df])
        full_df = full_df[~full_df.index.duplicated(keep='last')].sort_index()
    else:
        full_df = new_merged_df
    
    # Ensure column order matches training
    if 'Google_Trends' in full_df.columns:
        full_df.drop(columns=['Google_Trends'], inplace=True)
    
    expected_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'BTC_ETH_Ratio', 'BTC_Gold_Ratio', 'DXY', 'US10Y', 'RSI', 
        'Sentiment', 'News_Sentiment'
    ]
    
    # Validation 1: Ensure columns exist
    for col in expected_cols:
        if col not in full_df.columns:
            print(f"CRITICAL: Missing feature {col}. Initializing with 0.")
            full_df[col] = 0.0

    full_df = full_df[expected_cols]
    
    # Validation 2: Imputation instead of aggressive dropping
    initial_rows = len(full_df)
    full_df.ffill(inplace=True)
    full_df.bfill(inplace=True)
    full_df.fillna(0, inplace=True)
    
    final_rows = len(full_df)
    print(f"Data Imputation complete. Rows: {initial_rows} -> {final_rows}")
    
    if final_rows == 0:
        raise ValueError("Data Integrity Failure: Dataset is empty after processing.")
        
    # Save the updated dataset locally
    full_df.to_csv(local_path)
    
    # Create a Cleaned Version for Model Training/Inference (Drop partial today)
    clean_df = full_df.copy()
    if len(clean_df) > 1:
        clean_df = clean_df.iloc[:-1]
    
    # Validation 3: Minimum sample check for LSTM
    min_required = cloud_config.LOOKBACK_DAYS + cloud_config.FORECAST_DAYS + 10
    if len(clean_df) < min_required:
        print(f"WARNING: Dataset too small ({len(clean_df)} rows). LSTM requires at least {min_required}.")
    
    print(f"Dataset updated. Full: {full_df.shape}, Clean: {clean_df.shape}")
    
    return full_df, clean_df

def create_sequences(scaled_data, lookback=cloud_config.LOOKBACK_DAYS, forecast=cloud_config.FORECAST_DAYS):
    """Create multi-step sequences for LSTM."""
    X, y = [], []
    for i in range(len(scaled_data) - lookback - forecast + 1):
        X.append(scaled_data[i : i + lookback])
        y.append(scaled_data[i + lookback : i + lookback + forecast, 3])
    return np.array(X), np.array(y)

def get_last_hour_price_with_cache():
    """Returns the close price of the last finished hour, using a local cache to minimize API calls."""
    cache_path = os.path.join("data", "hourly_cache.json")
    now = datetime.now()
    
    # Check Cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
                cache_dt = datetime.fromisoformat(cache['timestamp'])
                # If the cache is from the current hour or later, it's valid for the PREVIOUS finished hour
                if cache_dt.hour == now.hour and cache_dt.date() == now.date():
                    return cache['price']
        except:
            pass
            
    # API Fetch
    try:
        data = yf.download("BTC-USD", period="1d", interval="1h", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if len(data) >= 2:
            # iloc[-2] is the last completed hour
            price = float(data['Close'].iloc[-2])
            
            # Save Cache
            os.makedirs("data", exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump({"timestamp": now.isoformat(), "price": price}, f)
            return price
    except Exception as e:
        print(f"Hourly API Error: {e}")
        
    return None

if __name__ == "__main__":
    full_df, clean_df = prepare_merged_dataset()
    print(full_df.tail())
