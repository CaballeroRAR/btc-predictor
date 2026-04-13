# BTC Predictor: Production MLOps Infrastructure

## Technical Overview
The BTC Predictor is a serverless MLOps system deployed on Google Cloud Platform (GCP). It utilizes a stacked Long Short-Term Memory (LSTM) architecture with Monte Carlo Dropout for uncertainty estimation. The project is designed for autonomous price trajectory projection using 14 distinct feature channels across market, network, and macro-economic metrics.

## Architecture
The system utilizes a decoupled three-tier architecture in the `us-central1` region:

1.  **Orchestration Layer (Cloud Run)**: Streamlit-based UI for data visualization and coordination tracking.
2.  **Autonomous Worker (Cloud Run)**: FastAPI-based execution environment for hourly drift recalibration and retraining triggers.
3.  **Neural Compute (Vertex AI)**: Custom training jobs triggered via the Vertex SDK on high-availability CPU instances.
4.  **Persistence Layer**: Synchronized storage using GCS for model artifacts and a dedicated Firestore instance (`btc-pred-db`) for state management.

## MLOps Pipeline Stabilization (Technical Resolutions)

### Distributed Identity and Authentication
- **Problem**: 403 Forbidden/Unauthorized errors during Cloud Scheduler-to-Worker invocation.
- **Solution**: Implemented OIDC-authenticated handshakes by granting the `roles/iam.serviceAccountTokenCreator` role to the Cloud Scheduler Service Agent.
- **Outcome**: Established a secure, automated handshake for all scheduled jobs.

### Resource and Persistence Realignment
- **Compute Scaling**: Increased Cloud Run memory allocation to **2Gi** and extended the `attempt-deadline` to **600s (10 minutes)** to accommodate compute-intensive Monte Carlo Dropout calculations.
- **Database Topology**: Redirected the `FirestoreRepository` from the generic `(default)` database to the project-specific **`btc-pred-db`** instance.
- **Storage Strategy**: Implemented `roles/storage.admin` project-level permissions for the `btc-forecaster-sa` to resolve bucket metadata (storage.buckets.get) denial errors during training job submission.

### Quota-Aware Neural Scheduling
- **Constraint**: Vertex AI GPU (NVIDIA Tesla T4) quota limitations causing 429 Submission Rejected errors.
- **Optimization**: Reconfigured the logic in `vertex_trigger.py` and `cloud_config.py` to target **`n1-standard-4` (CPU)** instances. 
- **Efficiency**: LSTM training for the 14-feature input tensor is completed within the standard Project Quota window, ensuring 100% nightly retraining availability.

## Neural Schema: 14-Feature Input Tensor
The engine processes a 60-day lookback window. Features are normalized via a `MinMaxScaler`.

| Index | Feature | Metric Category | Technical Role |
| :--- | :--- | :--- | :--- |
| 0-4 | O/H/L/C/V | Market Core | Primary price action and liquidity confirmation. |
| 5 | BTC/ETH | Network Alpha | Internal asset-class risk sentiment proxy. |
| 6-8 | Gold/DXY/US10Y | Macro Gravity | Global dollar-strength and risk-free opportunity cost. |
| 9-11 | RSI/Fear&Greed/NLP | Sentiment/Technical | VADER Sentiment Reasoning and crowd psychology. |
| 12-13 | Drift Alignment/Vol | Feedback Loop | Model self-correction based on mean percentage error. |

## Production Specifications
- **Project Context**: `btc-predictor-492515`
- **Region**: `us-central1`
- **Firestore Instance**: `btc-pred-db`
- **Runtime Environment**: `python-3.11-slim`
- **Core Stacks**: `TensorFlow 2.16.1` / `Keras 3.12.1` / `FastAPI` / `Streamlit`
- **Service Account**: `btc-forecaster-sa` (PoLP Configured)
