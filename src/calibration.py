import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import cloud_config as cloud_config

def calculate_daily_drift(model, scaler, input_data, actual_price, iterations=50, lr=0.01):
    """
    Find the sentiment drift (offset) that minimizes the error for yesterday's price.
    Returns: The optimal sentiment offset (in scaled units).
    """
    # 1. Prepare Input
    scaled_input = scaler.transform(input_data.values)
    X_orig = np.expand_dims(scaled_input, axis=0)
    X_orig = tf.convert_to_tensor(X_orig, dtype=tf.float32)
    
    # 2. Target Price (Scaled)
    # To get the goal price in scaled units for index 3 (Close)
    num_features = scaler.n_features_in_
    dummy = np.zeros((1, num_features))
    dummy[0, 3] = actual_price
    target_price_scaled = scaler.transform(dummy)[0, 3]
    
    # 3. Optimization Variable: Drift for the Sentiment column (Index 10)
    drift = tf.Variable(0.0, trainable=True, dtype=tf.float32)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    # Keras 3 Stability: Define a static call for gradient calculation
    @tf.function(reduce_retracing=True)
    def call_model(X_in):
        return model(X_in, training=False)
    
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            # Sentiment is the 11th column (index 10 in our 12-feature pipeline)
            mask = np.zeros((1, cloud_config.LOOKBACK_DAYS, num_features))
            mask[:, :, 10] = 1.0 
            mask = tf.cast(mask, tf.float32)
            
            X_calib = X_orig + (drift * mask)
            
            # Predict only the first step (Tomorrow's Price)
            preds = call_model(X_calib)
            pred_price = preds[0, 0] # Index 0 of the 30-day forecast
            
            loss = tf.square(pred_price - target_price_scaled)
            
        gradients = tape.gradient(loss, [drift])
        optimizer.apply_gradients(zip(gradients, [drift]))
        
    return drift.numpy()

def batch_calibrate_sentiment(model, scaler, full_df, depth=30):
    """
    Calculate daily drifts for the last 'depth' days.
    Returns a dataframe with Actual Sentiment vs Market-Aligned Sentiment.
    """
    lookback = cloud_config.LOOKBACK_DAYS
    calibration_dates = full_df.index[-depth:]
    results = []
    
    # We calibrate each day based on the price that occurred that day
    # using the data available at that time (the lookback window).
    for i in range(len(full_df) - depth, len(full_df)):
        window = full_df.iloc[i - lookback : i]
        actual_price = full_df.iloc[i]['Close']
        actual_sentiment = full_df.iloc[i-1]['Sentiment'] # Real-world signal
        
        drift_scaled = calculate_daily_drift(model, scaler, window, actual_price)
        
        # Convert drift back to 'Sentiment Points' (0-100 scale)
        # For MinMaxScaler: scaled = (raw - min) / (max - min)
        # Therefore: delta_raw = delta_scaled * (max - min)
        sentiment_index = 10 # Fear & Greed Index
        sentiment_range = scaler.data_range_[sentiment_index]
        drift_points = drift_scaled * sentiment_range
        
        # Market Aligned Sentiment = Signal + Optimal Drift
        # CLIP to valid range [0, 100] to prevent out-of-distribution inputs
        market_aligned = np.clip(actual_sentiment + drift_points, 0, 100)
        
        results.append({
            "Date": full_df.index[i],
            "Actual_Sentiment": actual_sentiment,
            "Market_Aligned": market_aligned,
            "Drift": market_aligned - actual_sentiment # Resulting effective drift
        })
        
    return pd.DataFrame(results).set_index("Date")
