import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("core.analysis")

# --- Feature Group Mapping (Indices in the 12-feature schema) ---
FEATURE_GROUPS = {
    "Network Stats": [5],            # BTC_ETH_Ratio
    "Gravity Assets": [6, 7, 8],     # BTC_Gold_Ratio, DXY, US10Y
    "Psychology": [10, 11]           # Sentiment, Wikipedia interest
}

def calculate_signal_impact(model, scaler, recent_data, base_mean, strategy):
    """
    Automated Alpha Signal Evaluation:
    Measures the USD impact of each group by comparing Baseline to Ablated forecasts.
    """
    impact_data = []
    # Using the standardized strategy pattern for consistency
    SHADOW_ITERATIONS = 20 

    for group_name, indices in FEATURE_GROUPS.items():
        # Perform MC-Inference with feature ablation
        ablated_mean, _ = strategy.predict(
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
        
    logger.info("Signal impact attribution completed successfully.")
    return pd.DataFrame(impact_data)
