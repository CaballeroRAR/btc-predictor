import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from src.core.engine.lstm_strategy import LSTMMonteCarloStrategy
from src.core.standardizer import MarketStandardizer
from src.repositories.firestore_repo import FirestoreRepository
from src.repositories.prediction_repo import PredictionRepository
from src.repositories.calibration_repo import CalibrationRepository
from src.repositories.asset_repo import AssetRepository
from src.utils.logger import setup_logger
import cloud_config as cloud_config
from src.core.analysis import calculate_signal_impact
from data_loader import get_last_hour_price_with_cache

logger = setup_logger("facades.forecasting")

class ForecastingFacade:
    """
    The main architectural entry point for all Bitcoin forecasting operations.
    Orchestrates data cleaning, sentiment calibration, inference, and persistence.
    """
    def __init__(self):
        self.strategy = LSTMMonteCarloStrategy()
        self.standardizer = MarketStandardizer()
        self.prediction_repo = PredictionRepository()
        self.calibration_repo = CalibrationRepository()
        self.firestore_repo = FirestoreRepository()
        self.assets = AssetRepository()

    def get_forecast(self, model, scaler, clean_df, force=False, source="USER", include_impact=True):
        """
        Executes the full forecasting pipeline with caching.
        """
        # 1. Cache Layer
        if not force:
            snapshot = self.firestore_repo.get_latest_snapshot()
            if snapshot:
                logger.info("Serving high-speed forecast from Firestore cache")
                return self._format_snapshot(snapshot)

        logger.info(f"Initiating Ground-Up Forecast (Source: {source})")

        # 2. Data Standardization
        clean_df = self.standardizer.enforce_schema(clean_df)

        # 3. Sentiment Calibration
        prev_state = self.calibration_repo.load_latest_state()
        prev_drift = prev_state.get('drift_value', 0.0) if prev_state else 0.0
        
        # Cross-reference with last hourly price
        hourly_price = get_last_hour_price_with_cache()
        reference_price = hourly_price if hourly_price else clean_df['Close'].iloc[-1]

        # Calculate new drift using native calibration logic
        drift_df = self._calibrate_market_drift(model, scaler, clean_df)
        avg_drift = drift_df['Drift'].mean() if not drift_df.empty else prev_drift
        
        # Persist calibration state
        self.calibration_repo.save_state(avg_drift, reference_price, cloud_config.MODEL_PATH)

        # 4. Strategy Execution (Inference)
        recent_data = clean_df.tail(cloud_config.LOOKBACK_DAYS).copy()
        
        # Apply market-aligned psychology shifts
        recent_data['Sentiment'] = np.clip(recent_data['Sentiment'] + avg_drift, 0, 100)
        recent_data['Google_Trends'] = np.clip(recent_data['Google_Trends'] + avg_drift/2, 0, 100)

        # Execute Monte Carlo Inference
        mean, std = self.strategy.predict(model, scaler, recent_data)
        tight_std = std * 0.5  # Industrial confidence interval

        # 5. Timeline & Backtest Execution
        # (Migrated from legacy_engine.get_backtest_predictions)
        backtest_series = self._generate_backtest(model, scaler, clean_df)
        forecast_start = clean_df.index[-1] + timedelta(days=1)
        dates = [forecast_start + timedelta(days=i) for i in range(cloud_config.FORECAST_DAYS)]

        # 6. Persistence & Internal Logging
        self.prediction_repo.log_prediction_batch(dates, mean)
        
        results = {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'prices': mean.tolist(),
            'std': tight_std.tolist(),
            'backtest_values': backtest_series.values.tolist(),
            'backtest_dates': [d.strftime('%Y-%m-%d') for d in backtest_series.index],
            'avg_drift': avg_drift
        }
        self.firestore_repo.save_system_snapshot(results)

        # 7. UI / API Response Object
        res = {
            'is_cached': False,
            'calculation_time': datetime.now(timezone.utc),
            'dates': dates,
            'prices': mean,
            'std': tight_std,
            'backtest': backtest_series,
            'avg_drift': avg_drift
        }
        
        # 8. Optimized Signal Attribution (Conditional)
        if include_impact:
            res['impact_df'] = calculate_signal_impact(model, scaler, recent_data, mean, self.strategy)
            
        return res

    def _generate_backtest(self, model, scaler, full_df, depth=30):
        """
        Produce a serial backtest for the last N days to evaluate model fitness.
        Migrated from forecasting_engine.get_backtest_predictions.
        """
        import tensorflow as tf
        lookback = cloud_config.LOOKBACK_DAYS
        backtest_dates = full_df.index[-depth:]
        preds = []
        
        for i in range(len(full_df) - depth, len(full_df)):
            window = full_df.iloc[i - lookback : i]
            scaled = scaler.transform(window.values)
            X = tf.convert_to_tensor(np.expand_dims(scaled, axis=0), dtype=tf.float32)
            
            # Simple inference for backtest (no dropout needed)
            p_scaled = model(X, training=False)[0, 0]
            
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, 3] = p_scaled
            preds.append(scaler.inverse_transform(dummy)[0, 3])
            
        return pd.Series(preds, index=backtest_dates)

    def sync_market_actuals(self, full_df):
        """Matches past predictions with actual market action."""
        logger.info("Initiating Market Actuals Synchronization")
        if full_df.empty:
            return 0
            
        recent_actuals = full_df['Close'].iloc[:-1].tail(7)
        updated_count = 0
        for date, price in recent_actuals.items():
            count = self.prediction_repo.update_actual_price_matching(date.strftime("%Y-%m-%d"), price)
            updated_count += count
        
        return updated_count

    def get_performance_history(self):
        """
        Retrieves historical prediction performance as a UI-ready DataFrame.
        Maintains contract parity with the legacy UI blocks.
        """
        history = self.prediction_repo.get_history()
        if not history:
            return pd.DataFrame()
            
        df = pd.DataFrame(history)
        # Rename or ensure repository schema matches UI expectations
        # Expected index is Date/forecast_date
        if 'forecast_date' in df.columns:
            df['Date'] = pd.to_datetime(df['forecast_date'])
            df.set_index('Date', inplace=True)
            
        if 'predicted_price' not in df.columns and 'price' in df.columns:
            df = df.rename(columns={'price': 'predicted_price'})
            
        # UI Chart expects 'actual_price' to be filled for historical metrics
        # The prediction_repo handles this matching during sync_market_actuals
        return df


    def _calibrate_market_drift(self, model, scaler, clean_df, depth=3):
        """
        Native implementation of market alignment.
        Calculates the mean prediction error (drift) over the last N days.
        """
        logger.info(f"Targeting Market Alignment (Depth: {depth} days)...")
        results = []
        
        # Calculate drift over the last 'depth' days
        for i in range(depth, 0, -1):
            try:
                # Target date (T) and input window (T-lookback)
                idx = len(clean_df) - i
                if idx < cloud_config.LOOKBACK_DAYS: continue
                
                target_date = clean_df.index[idx]
                actual_price = clean_df['Close'].iloc[idx]
                
                # Window for prediction
                window = clean_df.iloc[idx - cloud_config.LOOKBACK_DAYS : idx]
                pred_mean, _ = self.strategy.predict(model, scaler, window)
                
                drift = actual_price - pred_mean
                # Normalize drift relative to Sentiment/Trends scale (roughly)
                # This is a heuristic to convert USD error to a '心理' (psychological) offset
                norm_drift = np.sign(drift) * min(abs(drift) / 500, 10.0) 
                
                results.append({
                    'Date': target_date,
                    'Actual': actual_price,
                    'Predicted': pred_mean,
                    'Drift': norm_drift
                })
            except Exception as e:
                logger.warning(f"Calibration step failed for index -{i}: {e}")
                
        return pd.DataFrame(results)

    def _format_snapshot(self, snapshot):
        """Helper to convert Firestore snapshot into runtime-ready dict."""
        return {
            **snapshot,
            'is_cached': True,
            'calculation_time': snapshot.get('timestamp'),
            'dates': [pd.to_datetime(d) for d in snapshot['dates']],
            'prices': np.array(snapshot['prices']),
            'std': np.array(snapshot['std']),
            'backtest': pd.Series(snapshot['backtest_values'], index=pd.to_datetime(snapshot['backtest_dates'])),
        }
