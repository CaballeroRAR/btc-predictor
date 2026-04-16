import os
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
from src import cloud_config
from src.core.analysis import calculate_signal_impact
from src.core.data_orchestrator import data_orchestrator

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
        # 1. Cache Layer with Temporal Integrity Guard
        if not force:
            snapshot = self.firestore_repo.get_latest_snapshot()
            if snapshot:
                # Verify that the cached forecast is still chronologically relevant
                first_date = pd.to_datetime(snapshot['dates'][0]).date()
                today_dt = datetime.now().date()
                if first_date >= today_dt:
                    logger.info("Serving current high-speed forecast from Firestore")
                    return self._format_snapshot(snapshot)
                else:
                    logger.warning(f"Cached forecast is chronologically stale (Started {first_date}). Triggering fresh alignment.")

        logger.info(f"Initiating Ground-Up Forecast (Source: {source})")

        # 2. Data Standardization
        clean_df = self.standardizer.enforce_schema(clean_df)

        # 3. Sentiment Calibration
        prev_state = self.calibration_repo.load_latest_state()
        prev_drift = prev_state.get('drift_value', 0.0) if prev_state else 0.0
        
        # Cross-reference with last hourly price
        live_ctx = self.get_live_market_context()
        hourly_price = live_ctx['live_price']
        reference_price = hourly_price if hourly_price else clean_df['Close'].iloc[-1]

        # Calculate new drift using native calibration logic
        drift_df = self._calibrate_market_drift(model, scaler, clean_df)
        avg_drift = drift_df['Drift'].mean() if not drift_df.empty else prev_drift
        
        # Persist calibration state
        self.calibration_repo.save_state(avg_drift, reference_price, cloud_config.MODEL_PATH)

        # 4. Strategy Execution with Pulse Injection
        # We inject Today's live price into the lookback window
        # to ensure the first prediction is a session-close estimate.
        injected_df = self._inject_intraday_pulse(clean_df, live_ctx, avg_drift)
        recent_data = injected_df.tail(cloud_config.LOOKBACK_DAYS).copy()

        # Execute Monte Carlo Inference
        # Returns (Batch, Time, features) or similar - we need the first forecast day
        m_raw, s_raw = self.strategy.predict(model, scaler, recent_data)
        
        # Refinement: Global Scalar Hardening
        # Ensures mean[0] and std[0] are always scalars regardless of model architecture
        mean = np.array(m_raw).flatten()
        std = np.array(s_raw).flatten()
        tight_std = std * 0.5
        
        # 5. Timeline & Inception Grounding (Refinement F-03)
        today_dt = datetime.now().date()
        
        # Neural Bias: Absolute % difference between Model Observation and Reality
        neural_bias_pct = abs(reference_price - mean[0]) / reference_price
        
        # Volatility Metric: Use the 1st day MC Standard Deviation as a % of price
        expected_variance = float((tight_std[0] / mean[0]) * 100)
        
        # --- MOMENTUM-FAVORED GROUNDING LOGIC ---
        # User decision: Favor Neural Model during high volatility.
        # Implementation: Decrease GROUNDING_FACTOR as variance increases.
        BASE_G = float(os.getenv("FORECAST_GROUNDING_FACTOR", 0.5))
        
        # If variance > 2.5%, we start scaling down the anchor to favor momentum
        volatility_threshold = 2.5 
        if expected_variance > volatility_threshold:
            # Decay the grounding factor: deeper uncertainty = higher model autonomy
            adaptive_g = BASE_G * np.exp(-(expected_variance - volatility_threshold) / 5.0)
            adaptive_g = max(adaptive_g, 0.1) # Floor at 10% to prevent full disconnection
        else:
            adaptive_g = BASE_G

        initial_alignment = (reference_price * adaptive_g) + (mean[0] * (1 - adaptive_g))
        shift = initial_alignment - mean[0]
        
        # --- NEURAL REACTIVITY AUDIT ---
        if len(mean) >= 7:
            growth_7d = ((mean[6] - mean[0]) / mean[0]) * 100
        else:
            growth_7d = 0.0 # Fallback for single-point forecasts
            
        logger.info("\n" + "="*40)
        logger.info("--- NEURAL REACTIVITY AUDIT (REFINED) ---")
        logger.info(f"Target Period:  {today_dt}")
        logger.info(f"Live Price:     ${float(reference_price):,.2f}")
        logger.info(f"Neural Obs:     ${float(mean[0]):,.2f}")
        logger.info(f"Model Variance: {float(expected_variance):.2f}%")
        logger.info(f"Adaptive G:     {float(adaptive_g):.2f} (Favor: {'MODEL' if adaptive_g < BASE_G else 'PRICE'})")
        logger.info(f"Neural Bias:    ${float(reference_price - mean[0]):,.2f} ({float(neural_bias_pct)*100:+.2f}%)")
        logger.info(f"7-Day Momentum: {float(growth_7d):+.2f}%")
        logger.info("="*40 + "\n")
        
        # REFINEMENT: Confidence-Weighted Decay
        x_steps = np.linspace(0, 10, len(mean))
        # Logistic decay: slower at first, then drops off to 0
        decay = 1 / (1 + np.exp(x_steps - 5)) 
        # Normalize decay to start at 1.0
        decay = decay / decay[0]
        
        # Apply the grounded shift to the entire forecast horizon
        mean = mean + (shift * decay)

        # Every point in 'mean' is now a grounded neural prediction for [Today, Tomorrow, ...]
        all_dates = [datetime.combine(today_dt, datetime.min.time()) + timedelta(days=i) for i in range(len(mean))]
        
        dates = all_dates
        mean = mean
        tight_std = tight_std

        # 6. Persistence & Internal Logging
        backtest_series = self._generate_backtest(model, scaler, clean_df)
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
                
                drift = float(actual_price) - float(pred_mean[0])
                # Normalize drift relative to Sentiment/Trends scale (roughly)
                # This is a heuristic to convert USD error to a psychological offset
                norm_drift = np.sign(drift) * min(abs(drift) / 500.0, 10.0) 
                
                results.append({
                    'Date': target_date,
                    'Actual': actual_price,
                    'Predicted': pred_mean,
                    'Drift': norm_drift
                })
            except Exception as e:
                logger.warning(f"Calibration step failed for index -{i}: {e}")
                
        return pd.DataFrame(results)

    def get_live_market_context(self):
        """
        Orchestrates real-time tactical overlays for the UI.
        Encapsulates Live price and Curiosity Pulse logic.
        """
        live_price = data_orchestrator.adapter.fetch_price_data(years=1/365).iloc[-1]['Close']
        wiki_pulse = data_orchestrator.adapter.fetch_hourly_views()
        
        interest_pulse = 0.0
        if not wiki_pulse.empty:
            interest_pulse = float(wiki_pulse['Curiosity_Hourly'].iloc[-1])
            
        return {
            "live_price": live_price,
            "interest_pulse": interest_pulse,
            "timestamp": datetime.now(timezone.utc)
        }

    def _inject_intraday_pulse(self, df, live_ctx, drift):
        """
        Synthesizes a 'Live' candle for Today and appends it to history.
        Ensures the model captures the $6,000 breakout currently in flight.
        """
        today_dt = datetime.now().date()
        if df.index[-1].date() == today_dt:
            return df # Already has today's finalized data
            
        last_row = df.iloc[-1].copy()
        live_price = live_ctx['live_price']
        
        # Build Pulse Row
        pulse_row = last_row.copy()
        pulse_row.name = pd.Timestamp(today_dt)
        
        # Inject Price Pulse
        pulse_row['Open'] = last_row['Close']
        pulse_row['High'] = max(pulse_row['Open'], live_price)
        pulse_row['Low'] = min(pulse_row['Open'], live_price)
        pulse_row['Close'] = live_price
        
        # Inject Psychology Pulse
        pulse_row['Sentiment'] = np.clip(last_row['Sentiment'] + drift, 0, 100)
        pulse_row['Google_Trends'] = np.clip(last_row['Google_Trends'] + drift/2, 0, 100)
        
        logger.info(f"INJECTION: Appending {today_dt} pulse (${live_price:,.2f}) to Neural Window")
        return pd.concat([df, pd.DataFrame([pulse_row])])

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

