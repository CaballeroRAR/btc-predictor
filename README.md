# BTC Predictor Project: v2.1 Industrial Infrastructure

The BTC Predictor is a professional-grade forecasting system that utilizes a stacked Long Short-Term Memory (LSTM) architecture to project Bitcoin price trajectories. The system is built on a serverless Google Cloud Platform (GCP) backbone, integrating high-frequency market data with institutional macro-economic indicators.

---

## Technical Architecture

The project implements a Decoupled Three-Tier Architecture to ensure institutional stability:

1.  **Orchestration Layer**: Streamlit-based interface deployed on Cloud Run, providing Monte Carlo uncertainty estimation and precision coordinate tracking.
2.  **Autonomous Worker Layer**: FastAPI service on Cloud Run managing hourly drift recalibration and automated pipeline triggers via Cloud Scheduler.
3.  **Neural Compute Layer**: Custom training environment on Vertex AI using Spot Provisioning Models for cost-efficient model lifecycle management.

---

## Granular Neural Schema: The 14-Feature Input Tensor

The forecasting engine processes a 60-day historical window across 14 distinct feature channels. Each feature is normalized via a MinMaxScaler to maintain gradient stability during backpropagation.

### Group 1: Market Core (Indices 0-4)
- **0. Open**: Daily opening price in USD.
- **1. High**: Daily relative peak. Used by the model to quantify intraday volatility.
- **2. Low**: Daily support floor.
- **3. Close**: The primary regression target.
- **4. Volume**: Market liquidity signal. Used to confirm the strength of price movements.

### Group 2: Network Alpha (Index 5)
- **5. BTC/ETH Ratio**: Proxy for blockchain-native risk sentiment. A rising ratio indicates a "Flight to Quality" or risk-off sentiment within the cryptocurrency asset class.

### Group 3: Macro Gravity Engine (Indices 6-8)
- **6. BTC/Gold Ratio**: Measures the valuation of "Digital Gold" against physical sound money. Captures shifts in the global store-of-value narrative.
- **7. USD Index (DXY)**: The primary currency headwind. Quantifies dollar-strength pressure on Bitcoin's USD valuation.
- **8. US 10-Year Treasury Yield (US10Y)**: Represents the opportunity cost of risk-off yielding assets. Higher risk-free rates historically exert downward pressure on speculative assets.

### Group 4: Psychological Velocity (Indices 9-11)
- **9. RSI (14-Day)**: Relative Strength Index. Identifies overbought or oversold technical conditions.
- **10. Fear & Greed Index**: Aggregated market sentiment (Alternative.me API). Quantifies crowd psychology extremes.
- **11. News Sentiment (VADER NLP)**: Compound sentiment scores derived from CoinDesk and CoinTelegraph RSS feeds. Uses Valency Aware Dictionary for Sentiment Reasoning to map tone to a 0-100 scale.

### Group 5: Closed-Loop System Feedback (Indices 12-13)
- **12. Drift Alignment**: The mean percentage error between the previous forecast and actual market close. This feature allows the model to "self-correct" based on its own recent performance.
- **13. Drift Volatility**: The standard deviation of recent prediction errors. Provides the model with a "Confidence Context" based on historical performance variance.

---

## Data Integrity and Resilience

The pipeline implements Selective Imputation to ensure high availability:

1.  **Imputation Logic**: In the event of an API failure (e.g., DXY or News Feed lockout), the system performs forward and backward filling (`ffill` and `bfill`) followed by zero-filling. This prevents empty datasets from crashing the training or inference cycles.
2.  **Validation Gate**: A minimum sample check ensures that the dataset contains at least (Lookback + Forecast + 10) rows before proceeding to the Neural Layer.

---

## Deployment and MLOps

### Automated Validation (Command Shadowing)
To ensure deployment safety without live cloud interaction, the system uses a Command Shadowing test suite. This suite intercepts `gcloud` and `gsutil` calls to verify logic and configurations against a predefined "Operation Log."

### Specialized Agent Workflows
The project is maintained using agentic workflows located in `.agent/workflows/`. These include:
- `analyze-project`: Structural audit and architectural mapping.
- `fix-pipelines`: Diagnostic and healing logic for inference failures.
- `mlops-expert`: Scalability and security audit for GCP infrastructure.

---

## Environment Configuration
- **Project ID**: `btc-predictor-492515`
- **Region**: `us-central1`
- **Runtime**: `python-3.11-slim`
- **Neural Stack**: `keras==3.12.1` / `tensorflow==2.16.1`
