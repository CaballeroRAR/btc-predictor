import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import cloud_config as cloud_config

# --- Feature Group Mapping ---
FEATURE_GROUPS = {
    "Network Stats": [5],            # BTC_ETH_Ratio
    "Gravity Assets": [6, 7, 8],     # BTC_Gold_Ratio, DXY, US10Y
    "Psychology": [10, 11]           # Fear & Greed, News Sentiment
}

def calculate_withdrawal_date(dates, prices, target_price):
    """Find the first date where price meets or exceeds target."""
    for i, p in enumerate(prices):
        if p >= target_price:
            return dates[i]
    return None

def predict_with_uncertainty(model, scaler, current_data, iterations=20, ignored_indices=None):
    """
    Predict next 30 days using Monte Carlo Dropout.
    Supports feature ablation by replacing ignored indices with their column-wise mean.
    """
    data_values = current_data.values.copy()
    
    if ignored_indices:
        # Replace ignored features with their mean over the lookback window
        for idx in ignored_indices:
            data_values[:, idx] = np.mean(data_values[:, idx])
            
    scaled_input = scaler.transform(data_values)
    X = np.expand_dims(scaled_input, axis=0)
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    
    # Perform multiple passes with dropout active
    preds_list = []
    
    @tf.function(reduce_retracing=True)
    def mc_step(X_batch):
        return model(X_batch, training=True)

    for _ in range(iterations):
        preds_scaled = mc_step(X)
        
        # Inverse transform only the relevant closing price
        num_features = scaler.n_features_in_
        dummy = np.zeros((cloud_config.FORECAST_DAYS, num_features))
        dummy[:, 3] = preds_scaled[0]
        preds_unscaled = scaler.inverse_transform(dummy)[:, 3]
        preds_list.append(preds_unscaled)
        
    preds_list = np.array(preds_list)
    mean = preds_list.mean(axis=0)
    std = preds_list.std(axis=0) # 1-Sigma
    
    return mean, std

def calculate_signal_impact(model, scaler, recent_data, base_mean):
    """
    Automated Alpha Signal Evaluation:
    Measures the USD impact of each group by comparing Baseline to Ablated forecasts.
    """
    impact_data = []
    SHADOW_ITERATIONS = 20 

    for group_name, indices in FEATURE_GROUPS.items():
        ablated_mean, _ = predict_with_uncertainty(
            model, scaler, recent_data, iterations=SHADOW_ITERATIONS, ignored_indices=indices
        )
        
        full_window_mean = np.mean(base_mean)
        ablated_window_mean = np.mean(ablated_mean)
        
        usd_delta = full_window_mean - ablated_window_mean
        conviction = (abs(usd_delta) / full_window_mean) * 100
        
        impact_data.append({
            "Signal Group": group_name,
            "USD Impact": usd_delta,
            "Impact Magnitude": abs(usd_delta),
            "Relative Importance": conviction,
            "Direction": "Bullish Influence" if usd_delta > 0 else "Bearish Influence"
        })
        
    return pd.DataFrame(impact_data)

def get_backtest_predictions(model, scaler, full_df, depth=30):
    """
    Generate 1-day-ahead predictions for the last 'depth' days.
    """
    lookback = cloud_config.LOOKBACK_DAYS
    backtest_dates = full_df.index[-depth:]
    historical_preds = []
    
    @tf.function(reduce_retracing=True)
    def call_model(X_in):
        return model(X_in, training=False)

    for i in range(len(full_df) - depth, len(full_df)):
        window = full_df.iloc[i - lookback : i]
        scaled_window = scaler.transform(window.values)
        X = np.expand_dims(scaled_window, axis=0)
        pred_scaled = call_model(X)[0, 0] 
        
        num_features = scaler.n_features_in_
        dummy = np.zeros((1, num_features))
        dummy[0, 3] = pred_scaled
        pred_unscaled = scaler.inverse_transform(dummy)[0, 3]
        historical_preds.append(pred_unscaled)
        
    return pd.Series(historical_preds, index=backtest_dates)
