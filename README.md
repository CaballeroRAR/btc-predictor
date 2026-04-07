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

---

## Technical Deep-Dive: Data Science & Engineering

### 1. The Neural Architecture (Stacked LSTM)
The core forecasting engine is a **Sequential Recurrent Neural Network (RNN)** using two stacked **Long Short-Term Memory (LSTM)** layers (64 units each).
- **Temporal Memory**: LSTMs are used to capture long-range dependencies in market cycles (e.g., historical RSI rebounds vs. DXY trends).
- **Dropout Regularization**: We apply a $0.2$ Dropout rate between layers. This is critical for **Inference Science**, as it enables our Monte Carlo uncertainty simulations.
- **Activation & Optimizer**: We utilize **ReLU** for non-linearity and the **Adam** optimizer with a Mean Squared Error (MSE) loss function for regression stability.

### 2. Feature Engineering: The "Macro Gravity" Schema (12 Features)
We moved beyond simple price-action by integrating cross-asset proxies that mathematically influence Bitcoin's dollar-denominated value:
| Feature | Logic / "Gravity" |
| :--- | :--- |
| **OHLCV (5)** | Standard market liquidity and price discovery. |
| **BTC/ETH Ratio** | Risk-on/Risk-off proxy within the crypto ecosystem. |
| **BTC/Gold Ratio** | "Digital Gold" vs. Physical Gold parity. |
| **DXY (USD Index)** | Captures dollar-strength headwinds; inverse correlation to BTC. |
| **US10Y Yields** | Opportunity cost of risk-off yielding assets; global macro headwind. |
| **RSI (14-day)** | Momentum oscillator for overbought/oversold detection. |
| **Sentiment** | Institutional/Retail fear & greed index (Fear & Greed API). |
| **Google Trends** | Social interest and retail "hype" proxy. |

### 3. Inference Science: Uncertainty & Calibration
Unlike "black box" predictors, this system provides a probabilistic range:
- **Monte Carlo Dropout (Uncertainty)**: During inference, we keep Dropout layers **active** (`training=True`) and run 50 simultaneous forecasts. The standard deviation of these outputs forms our high-confidence channels.
- **Sentiment-Calibrated Drift**: A proprietary optimization layer. If the model is physically underperforming yesterday's price, we use **Gradient Descent** on the `Sentiment` input alone to find the "market-implied" sentiment drift. This recalibrates the green forecast line to match real-world market weight before projecting into the future.

### 4. Cloud Infrastructure
- **Vertex AI Custom Container**: Used for local/cloud training parity.
- **Cloud Run (Dashboard)**: A serverless environment for the Streamlit UI.
- **Auto-Sync Model Lifecycle**: On every "Cold Start," the dashboard verifies its local `.h5` model against the GCS bucket. If a newer model is detected in GCS, it automatically downloads and hot-swaps the weights in memory.

