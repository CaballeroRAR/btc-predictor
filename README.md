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
4.  **Decoupled 3-Tier Architecture**: The application has been refactored into a modular infrastructure (Inference Engine, UI Blocks, and Orchestrator) to ensure scalability and ease of maintenance.
5.  **Explainable AI (XAI) Integration**: v2.0 introduces Signal Attribution via Feature Ablation, allowing users to see the specific USD impact of Macro, Network, and Psychological factor groups.

---

## [PROCESS] Phase 4: Stability, Precision & XAI

Current system state (Phase 4) transforms the dashboard into a high-performance, persistent investment tracking and analysis platform:

1.  **Signal Attribution Engine (XAI)**: Using **Feature Ablation**, the model identifies the "appropriate weight" of factor groups. By replacing specific signals (Gravity, Network, or Psychology) with their 60-day mean, the engine quantifies their exact USD impact on the forecast.
2.  **Precision Pinning Visualization**: The Investment Journal now features high-accuracy coordinate tracking. Target markers ("Orig" and "Curr") are dynamically pinned to their respective forecast trajectories (Snapshot vs. Live) rather than a static price level.
3.  **High-Frequency Calibration**: Market sentiment drift is optimized daily via Gradient Descent to align model behavior with near-real-time psychological shifts.
4.  **Agent-Orchestrated MLOps**: The project is managed by a suite of specialized AI Agents (Dispatcher, Refactor Supervisor, Style Auditor) to maintain structural integrity and a Standardized Professional Tone (SPT).

1.  **Incremental Data Loading**: Refactored the data loader to check for existing local data and fetch only the "missing delta" (the few days since the last run). This reduces API overhead and significantly improves startup speed.
2.  **Dual-Layer Performance Tracking**:
    -   **Historical Accuracy (D-1)**: Compares yesterday's actual market close against the mean of all predictions made on the day before.
    -   **Session Predicted Mean (Today)**: Compares the mean of all simulations run so far today against the latest live Bitcoin price, featuring a detailed **Hover Tooltip** with individual heartbeat logs.
3.  **Decoupled Investment Planning**: The slow model inference (50 iterations) is now decoupled from the simulation logic. Adjusting "Profit Targets" or "Entry Prices" provides **instant feedback** on withdrawal dates without re-running the heavy model.
4.  **Manual & Auto-Drift Management**:
    -   **Auto-Run**: The system automatically calculates **Sentiment Drift** on dashboard startup to align the forecast with yesterday's market action.
    -   **Force Refresh**: A manual "Force Market Refresh" button in the sidebar clears the cache, fetches fresh data, and re-predicts the 30-day window on demand.
5.  **Persistent Investment Journal**: Tracking ROI and **Withdrawal Date Shifts**. If today's forecast moves your target date earlier or later, the Journal dynamically updates with success/warning notifications and precision-aligned milestone markers.

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

The deployment process is now fully automated. Run the orchestration script to sync models, build containers, and deploy the dashboard in one step:

```powershell
.\automate_deployment.ps1
```

---

### Phase 3: Cleanup & Standards
The system has been sanitized for production:
- **Feature Consistency**: Strict 12-feature architecture maintained across all layers.
- **Backend Stability**: Keras 3 Monte Carlo loop stabilized with `tf.function`.
- **Automated Calibration**: Market sentiment drift is calculated and applied before every forecast.
- **Cache Management**: Local caches (`__pycache__`) are excluded from builds via updated `.gcloudignore`.

---

## Technical Deep-Dive: Data Science & Engineering

### 1. The Neural Architecture (Stacked LSTM)
The core forecasting engine is a **Sequential Recurrent Neural Network (RNN)** using two stacked **Long Short-Term Memory (LSTM)** layers (64 units each).
- **Temporal Memory**: LSTMs capture long-range dependencies in market cycles by processing 60-day historical window sequences.
- **Dropout Regularization**: A 0.2 dropout rate is applied between layers. This is active during the **Monte Carlo (MC) Dropout** phase for uncertainty estimation.
- **Loss & Optimization**: Optimized via the **Adam** optimizer with Mean Squared Error (MSE) loss, localized on the closing price regression.

### 2. Feature Engineering: The "Macro Gravity" Schema (12 Features)
The system integrates 12 distinct signals to capture market "Gravity":
- **OHLCV**: Core liquidity and price action.
- **Network Sentiment**: BTC/ETH Ratio (Network risk proxy).
- **Gravity Assets**: BTC/Gold Ratio, USD Index (DXY), and US 10-Year Treasury Yields (Asset opportunity cost).
- **Technical/Social**: RSI (14-day), Fear & Greed Index, and VADER-analyzed RSS News Sentiment.

### 3. Modular System Architecture
The application has been refactored into a **Three-Tier Architecture** for scalability:
- **`forecasting_engine.py`**: The "Inference Engine." Handles MC Dropout, backtesting, and the new **Signal Attribution** math.
- **`ui_blocks.py`**: The "Presentation Layer." Modularized Streamlit components and Plotly visualization logic.
- **`dashboard.py`**: The "Orchestrator." Manages application lifecycle, session state, and user interaction flow.

### 4. Advanced Evaluation: Signal Attribution & Calibration
- **Signal Attribution Breakdown**: Using **Feature Ablation**, the model identifies the "appropriate weight" of factor groups. By replacing specific signals (Gravity, Network, or Psychology) with their 60-day mean, the engine quantifies their exact USD impact on the forecast.
- **Inference Science (MC Dropout)**: The model performs 50 simultaneous forecasts with active dropout to generate a probabilistic 0.5-sigma "Execution Channel."
- **Gradient-Descent Calibration**: The system self-aligns daily. It runs a localized optimization on the `Sentiment` feature to find the "Market-Aligned" weight, minimizing error relative to the previous day's close price.

### 5. Infrastructure & MLOps
- **Auto-Sync Model Lifecycle**: On cold start, the system verifies local models against **Google Cloud Storage (GCS)**, performing hot-swaps of weights (`.h5`) if a newer version exists.
- **Serverless Serving**: Deployed on **GCP Cloud Run** using a scale-to-zero Dockerized architecture for cost-efficiency.

