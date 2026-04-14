import pandas as pd
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
from src.utils.logger import setup_logger
from src import cloud_config

class IndustrialMarketAdapter:
    """
    Adapter for external market data sources (API-heavy logic).
    Encapsulates communication with yfinance, Wikimedia, and RSS feeds.
    """
    def __init__(self):
        self.logger = setup_logger("adapters.market")
        self.analyzer = SentimentIntensityAnalyzer()
        self.user_agent = 'BTCPredictorIndustrial/1.0 (contact: caballero.data.scientist@tutamail.com)'

    def fetch_price_data(self, years=cloud_config.YEARS_HISTORY):
        """Fetch historical BTC, ETH, Gold, and Oil data."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        self.logger.info(f"Fetching market assets since {start_date.date()}...")
        
        assets = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "Gold": "GC=F",
            "DXY": "DX-Y.NYB",
            "US10Y": "^TNX"
        }
        
        dfs = {}
        for name, ticker_cmd in assets.items():
            df = yf.download(ticker_cmd, start=start_date, interval="1d")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            dfs[name] = df

        btc = dfs["BTC"]
        df = btc[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Macro alignment logic
        gold_close = dfs["Gold"]['Close'].reindex(btc.index).ffill()
        dxy_close = dfs["DXY"]['Close'].reindex(btc.index).ffill()
        us10y_close = dfs["US10Y"]['Close'].reindex(btc.index).ffill()
        
        df['BTC_ETH_Ratio'] = btc['Close'] / dfs["ETH"]['Close']
        df['BTC_Gold_Ratio'] = btc['Close'] / gold_close
        df['DXY'] = dxy_close
        df['US10Y'] = us10y_close
        
        # Technical Signal: RSI (14-day)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def fetch_wikipedia_views(self, article="Bitcoin", years=cloud_config.YEARS_HISTORY):
        """Fetch historical Wikipedia pageviews."""
        self.logger.info(f"Fetching Wikimedia views for '{article}'...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        start_str = start_date.strftime("%Y%m%d00")
        end_str = end_date.strftime("%Y%m%d00")
        
        headers = {'User-Agent': self.user_agent}
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{article}/daily/{start_str}/{end_str}"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                items = response.json().get('items', [])
                if items:
                    df = pd.DataFrame(items)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H')
                    df = df.rename(columns={'timestamp': 'Date', 'views': 'Curiosity_Raw'})
                    df.set_index('Date', inplace=True)
                    
                    # Normalize (0-100)
                    v_min, v_max = df['Curiosity_Raw'].min(), df['Curiosity_Raw'].max()
                    df['Google_Trends'] = (df['Curiosity_Raw'] - v_min) / (v_max - v_min) * 100
                    return df[['Google_Trends']]
        except Exception as e:
            self.logger.error(f"Wikimedia API failure: {e}")
            
        return pd.DataFrame()

    def fetch_rss_sentiment(self):
        """Analyze latest news sentiment via RSS."""
        feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/"
        ]
        
        all_headlines = []
        for url in feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    if any(x in entry.title.lower() for x in ["bitcoin", "btc"]):
                        all_headlines.append(entry.title)
            except Exception as e:
                self.logger.warning(f"RSS failure ({url}): {e}")
                
        if not all_headlines:
            return 0.0
            
        scores = [self.analyzer.polarity_scores(h)['compound'] for h in all_headlines]
        return sum(scores) / len(scores)

    def fetch_fng_sentiment(self):
        """Fetch historical Fear & Greed Index."""
        url = "https://api.alternative.me/fng/?limit=0&format=json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()['data']
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='s')
                df['value'] = df['value'].astype(int)
                df = df[['timestamp', 'value']].rename(columns={'timestamp': 'Date', 'value': 'Sentiment'})
                df.set_index('Date', inplace=True)
                return df
        except Exception as e:
            self.logger.error(f"F&G Index failure: {e}")
            
        return pd.DataFrame()

    def fetch_blockchain_metrics(self):
        """Fetch Bitcoin network hashrate and difficulty (On-chain Pulse)."""
        self.logger.info("Fetching Bitcoin Network Fundamentals (Blockchain.com)...")
        # Base url for multiple stats
        base_url = "https://api.blockchain.info/charts"
        
        metrics = {
            "Hashrate": "hash-rate",
            "Difficulty": "difficulty"
        }
        
        dfs = []
        for name, slug in metrics.items():
            try:
                # Fetch extensive history to match price data (approx 6 years)
                url = f"{base_url}/{slug}?timespan=6years&format=json&cors=true"
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    values = resp.json().get('values', [])
                    if values:
                        m_df = pd.DataFrame(values)
                        m_df['x'] = pd.to_datetime(m_df['x'], unit='s')
                        m_df = m_df.rename(columns={'x': 'Date', 'y': name})
                        m_df.set_index('Date', inplace=True)
                        dfs.append(m_df)
            except Exception as e:
                self.logger.warning(f"Failed to fetch {name}: {e}")
        
        if not dfs:
            return pd.DataFrame()
            
        final_df = dfs[0]
        for next_df in dfs[1:]:
            final_df = final_df.join(next_df, how='outer')
            
        final_df.ffill(inplace=True)
        return final_df

    def fetch_hourly_views(self, article="Bitcoin"):
        """Fetch 48hr high-resolution curiosity pulse."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        start_str = start_date.strftime("%Y%m%d00")
        end_str = end_date.strftime("%Y%m%d23")
        
        headers = {'User-Agent': self.user_agent}
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents/{article}/hourly/{start_str}/{end_str}"
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                items = response.json().get('items', [])
                if items:
                    df = pd.DataFrame(items)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d%H')
                    df = df.rename(columns={'timestamp': 'Date', 'views': 'Curiosity_Hourly'})
                    df.set_index('Date', inplace=True)
                    return df[['Curiosity_Hourly']]
        except Exception as e:
            self.logger.error(f"Hourly views failure: {e}")
            
        return pd.DataFrame()
