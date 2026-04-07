# BTC Predictor Project

This system predicts Bitcoin prices and sentiment drift using a stacked LSTM model. It features a training pipeline (local/Vertex AI) and an interactive Streamlit dashboard (Cloud Run), now enhanced with **12-feature Macro Gravity** integration.

## Key Enhancements: Macro Gravity & Precision

The latest version (v2.0) introduces significant architectural improvements over the original 9-feature deployment:

1.  **12-Feature Macro Engine**: We successfully integrated three "Gravity" assets that historically influence Bitcoin's dollar-denominated value:
    *   **Gold (GC=F)** & **BTC/Gold Ratio**: Captures "Sound Money" flow into/out of Bitcoin.
    *   **USD Index (DXY)**: Captures dollar-strength headwinds/tailwinds.
    *   **US10Y (Treasury Yields)**: Captures the opportunity cost of risk-off yielding assets.
2.  **High-Confidence Channels (0.5-Sigma)**: The dashboard probability bands are now tightened to 0.5 standard deviations. This creates a focused, high-conviction "Execution Channel" that tracks the mean movement much more closely than standard 1-sigma bands.
3.  **Monte Carlo Uncertainty (50 Iterations)**: We increased the simulation count to 50 for smoother, more deterministic forecast trajectories.
4.  **Dynamic Scaling Engine**: The system is now architecture-agnostic. It automatically detects the feature count (9 vs 12) from the loaded `.pkl` scaler, preventing dimension-mismatch crashes.

---

## Architectural Principles & Solved Quirks

Our infrastructure has stabilized over standard GCP setups by solving these complex issues natively:
1. **Keras Deserialization Mismatch**: Keras 3 often embeds micro-version specific metadata. We eliminated these bugs by:
    - Strictly pinning `keras==3.12.1` across all environments.
    - Exporting models in the stable HDF5 format (`btc_lstm_model.h5`).
    - Using `compile=False` during loading for smooth inference.
2. **PyTrends Rate Limits**: Cloud Run IP addresses often hit Google Trends' HTTP 429 errors. We implemented a fallback system: if search trends are blocked, the system defaults to a baseline proxy value (`50.0`) instead of crashing the dashboard.
3. **Visual Plotting Alignment**: We implemented a "Seamless Connection" fix. Forecast arrays now prepend the current close price and date, ensuring the forecast line stems physically from the end of the historical data without a 1-day visual "jump."

---

## Environment Setup

1.  **Local Virtual Environment**:
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **GCP Project Configuration** (`src/cloud_config.py`):
    - **Project ID**: `btc-predictor-492515`
    - **Region**: `us-central1`

---

## Deployment Workflow

When you are ready to publish changes to GCP:

### 1. Synchronize Model Weights
Before building the dashboard, push your latest local training to GCS (Cloud Run fetches these on startup):
```powershell
gcloud storage cp models/btc_lstm_model.h5 gs://btc-predictor-492515_cloudbuild/models/btc_lstm_model.h5
gcloud storage cp models/scaler.pkl gs://btc-predictor-492515_cloudbuild/models/scaler.pkl
```

### 2. Build and Deploy Dashboard
```powershell
# 1. Build the image
gcloud builds submit --config cloudbuild.app.yaml .

# 2. Deploy to Cloud Run
gcloud run deploy btc-dashboard --image=gcr.io/btc-predictor-492515/btc-dashboard --region=us-central1 --project=btc-predictor-492515 --memory=2Gi
```
