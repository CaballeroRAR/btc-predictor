import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
import cloud_config as cloud_config

def build_lstm_model(input_shape):
    """
    Build a Stacked LSTM for multi-step price prediction.
    Input shape: (lookback_days, num_features)
    Output: forecast_days (predicted close prices)
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(cloud_config.FORECAST_DAYS)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == "__main__":
    # Test build
    model = build_lstm_model((cloud_config.LOOKBACK_DAYS, 9))
    model.summary()
