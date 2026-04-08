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
    
    for _ in range(iterations):
        with tf.GradientTape() as tape:
            # Sentiment is the 11th column (index 10 in our 12-feature pipeline)
            mask = np.zeros((1, cloud_config.LOOKBACK_DAYS, num_features))
            mask[:, :, 10] = 1.0 
            mask = tf.cast(mask, tf.float32)
            
            X_calib = X_orig + (drift * mask)
            
            # Predict only the first step (Tomorrow's Price)
            preds = model(X_calib, training=False)
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
        # Scaler transformation for Sentiment (Index 10):
        # Result = (Raw - Min) / (Max - Min)
        # So Drif_Raw = Drift_Scaled * (Max - Min)
        sentiment_range = scaler.data_range_[10]
        drift_points = drift_scaled * sentiment_range
        
        market_aligned = actual_sentiment - drift_points # If model need -drift to match price, then market is 'aligned' with shifted sentiment
        
        results.append({
            "Date": full_df.index[i],
            "Actual_Sentiment": actual_sentiment,
            "Market_Aligned": market_aligned,
            "Drift": -drift_points # Negative drift means market is more optimistic than signal
        })
        
    return pd.DataFrame(results).set_index("Date")
