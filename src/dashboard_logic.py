import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import forecasting_engine as engine
import prediction_logger as pred_log
import calibration as calib
import model_lifecycle as lifecycle
import cloud_config as cloud_config
from data_loader import get_last_hour_price_with_cache

def get_base_forecast(db_mgr, model, scaler, clean_df, force=False, source="USER"):
    """
    Standardized model forecasting service. 
    Handles Firestore snapshot caching and core analytical pipeline.
    Returns: dict containing UI-ready visualization data.
    """
    
    # 1. Attempt Cache Retrieval (Instant Dashboard Performance)
    if not force:
        snapshot = db_mgr.get_latest_snapshot()
        if snapshot:
            print(f"[{datetime.now()}] [CACHE] Loading high-speed snapshot from Firestore.")
            return {
                **snapshot,
                'is_cached': True,
                'calculation_time': snapshot.get('timestamp'),
                'dates': [pd.to_datetime(d) for d in snapshot['dates']],
                'prices': np.array(snapshot['prices']),
                'std': np.array(snapshot['std']),
                'backtest': pd.Series(snapshot['backtest_values'], index=pd.to_datetime(snapshot['backtest_dates'])),
                'impact_df': pd.read_json(snapshot['impact_df_json']) if snapshot.get('impact_df_json') else None
            }

    # 2. Market Alignment (Sentiment Calibration)
    prev_state = lifecycle.load_calibration_state(db_mgr=db_mgr)
    prev_sentiment_drift = prev_state.get('drift_value', 0.0) if prev_state else 0.0
    
    hourly_ref = get_last_hour_price_with_cache()
    latest_price = hourly_ref if hourly_ref else clean_df['Close'].iloc[-1]
    
    # Standardize column set to exactly 12 features (Macro Gravity Schema)
    expected_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'BTC_ETH_Ratio', 'BTC_Gold_Ratio', 'DXY', 'US10Y', 'RSI', 
        'Sentiment', 'Google_Trends'
    ]
    clean_df = clean_df[expected_cols]

    # Perform Sentiment Calibration (Aligns psychology with Market)
    drift_df = calib.batch_calibrate_sentiment(model, scaler, clean_df, depth=3) 
    avg_sentiment_drift = drift_df['Drift'].mean() if not drift_df.empty else prev_sentiment_drift
    lifecycle.save_calibration_state(avg_sentiment_drift, latest_price)

    # 4. Forecast Execution
    recent_data_raw = clean_df.tail(cloud_config.LOOKBACK_DAYS).copy()
    recent_data_aligned = recent_data_raw.copy()
    
    # Apply calibrated drift strictly to psychology signals
    recent_data_aligned['Sentiment'] = np.clip(recent_data_aligned['Sentiment'] + avg_sentiment_drift, 0, 100)
    recent_data_aligned['Google_Trends'] = np.clip(recent_data_aligned['Google_Trends'] + avg_sentiment_drift/2, 0, 100)
    
    # Raw Prediction (Base Model)
    raw_mean, _ = engine.predict_with_uncertainty(model, scaler, recent_data_raw, iterations=50)
    
    # Aligned Prediction (Market Adjusted)
    mean, std = engine.predict_with_uncertainty(model, scaler, recent_data_aligned, iterations=50)
    tight_std = std * 0.5
    
    # 4. Signal Impact (Ablation)
    impact_df = engine.calculate_signal_impact(model, scaler, recent_data_aligned, mean)

    # 5. Timeline Preparation
    forecast_start = clean_df.index[-1] + timedelta(days=1)
    forecast_dates = [forecast_start + timedelta(days=i) for i in range(cloud_config.FORECAST_DAYS)]
    backtest_series = engine.get_backtest_predictions(model, scaler, clean_df)
    
    # 6. Logging
    pred_log.log_predictions(forecast_dates, mean)
    
    # 7. Prepare and Save Snapshot
    res_to_save = {
        'dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'raw_prices': raw_mean.tolist(),
        'prices': mean.tolist(),
        'std': tight_std.tolist(),
        'backtest_values': backtest_series.values.tolist(),
        'backtest_dates': [d.strftime('%Y-%m-%d') for d in backtest_series.index],
        'avg_drift': avg_sentiment_drift,
        'impact_df_json': impact_df.to_json() if impact_df is not None else None
    }
    db_mgr.save_system_snapshot(res_to_save)
    
    # 7.5 Log Live Drift Snapshot (Dedicated Tactical Collection)
    live_price = clean_df['Close'].iloc[-1]
    live_drift_val = ((mean[0] / live_price) - 1) * 100
    db_mgr.log_live_drift(
        forecast_date=forecast_dates[0].strftime('%Y-%m-%d'),
        prediction=mean[0],
        market_price=live_price,
        drift_pct=live_drift_val,
        source=source
    )
    
    # 8. Return UI-ready results
    return {
        'is_cached': False,
        'calculation_time': datetime.now(timezone.utc),
        'dates': forecast_dates,
        'raw_prices': raw_mean,
        'prices': mean,
        'std': tight_std,
        'backtest': backtest_series,
        'avg_drift': avg_sentiment_drift,
        'impact_df': impact_df
    }

def calculate_withdrawal_plan(base_res, entry_price, profit_target, investment):
    """
    Pure logic for calculating investment withdrawal milestones.
    """
    target_val = entry_price * (1 + profit_target/100)
    w_date = engine.calculate_withdrawal_date(base_res['dates'], base_res['prices'], target_val)
    
    return {
        **base_res, 
        'target': target_val, 
        'target_pct': profit_target, 
        'entry_price': entry_price, 
        'investment': investment, 
        'withdrawal_date': w_date
    }
