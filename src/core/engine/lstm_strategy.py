import numpy as np
import tensorflow as tf
import pandas as pd
from src.core.engine.base import PredictionStrategy
from src import cloud_config

class LSTMMonteCarloStrategy(PredictionStrategy):
    """
    Implements a Monte Carlo Dropout strategy for LSTM-based price forecasting.
    This provides both a prediction mean and a confidence interval (standard deviation).
    """
    def __init__(self):
        super().__init__("core.engine.lstm_mc")

    def predict(self, model, scaler, data, iterations=50, ignored_indices=None):
        """
        Executes the MC-Dropout inference loop.
        """
        self.logger.info(f"Executing Monte Carlo Dropout with {iterations} iterations")
        
        data_values = data.values.copy()
        if ignored_indices:
            self.logger.warning(f"Feature ablation active for indices: {ignored_indices}")
            for idx in ignored_indices:
                data_values[:, idx] = np.mean(data_values[:, idx])

        # Preprocessing
        scaled_input = scaler.transform(data_values)
        X = np.expand_dims(scaled_input, axis=0)
        X = tf.convert_to_tensor(X, dtype=tf.float32)

        preds_list = []
        
        # Optimized TF function for inference
        @tf.function(reduce_retracing=True)
        def mc_step(X_batch):
            return model(X_batch, training=True)

        for i in range(iterations):
            if i % 10 == 0:
                self.logger.debug(f"Iteration {i}/{iterations} complete")
            
            preds_scaled = mc_step(X)
            
            # Inverse transform only the closing price (index 3)
            num_features = scaler.n_features_in_
            dummy = np.zeros((cloud_config.FORECAST_DAYS, num_features))
            dummy[:, 3] = preds_scaled[0]
            preds_unscaled = scaler.inverse_transform(dummy)[:, 3]
            preds_list.append(preds_unscaled)
            
        preds_list = np.array(preds_list)
        mean = preds_list.mean(axis=0)
        std = preds_list.std(axis=0)
        
        self.logger.info("Monte Carlo inference loop completed successfully")
        return mean, std
