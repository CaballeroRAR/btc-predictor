# Pipeline Interaction Audit & Mapping

This document maps the flow of data and control across the BTC Predictor's pipelines, identifying the code files responsible for triggering, processing, and receiving data.

## 1. Interaction Map

```mermaid
graph TD
    subgraph Triggers
        D[main_dashboard.py]
        T[main_trainer.py]
        V[vertex_trigger.py]
        W[main_worker.py]
    end

    subgraph Processing_Layer
        OR[DataOrchestrator]
        AD[IndustrialMarketAdapter]
        ST[MarketStandardizer]
    end

    subgraph Receivers
        CA[data/merged_data.csv]
        MO[models/model.h5]
        SC[models/scaler.pkl]
        AS[AssetRepository / GCS]
    end

    D -->|Calls| OR
    W -->|Calls| OR
    T -->|Calls| OR
    OR -->|Uses| AD
    OR -->|Uses| ST
    OR -->|Persists| CA
    T -->|Saves| MO
    T -->|Saves| SC
    T -->|Publishes| AS
    V -->|Triggers Cloud| T
```

## 2. Pipeline Definitions

### A. Ingestion (ETL)
*   **Triggers**: `main_dashboard.py` (On load or Force Refresh), `main_worker.py` (Periodic).
*   **Processor**: `src/core/data_orchestrator.py`.
*   **Receiver**: `data/merged_data.csv`.
*   **Context**: Fetches from yfinance, alternative.me, and Wikimedia.

### B. Training
*   **Triggers**: `main_trainer.py` (Local), `vertex_trigger.py` (Cloud).
*   **Processor**: `src/main_trainer.py`.
*   **Receiver**: `models/` directory / GCS Bucket.
*   **Context**: Fits `MinMaxScaler`, trains LSTM, and saves artifacts for inference.

### C. Inference
*   **Triggers**: `main_dashboard.py` (Prediction call).
*   **Processor**: `src/main_dashboard.py` (via `MC Dropout` logic).
*   **Receiver**: UI Elements.
*   **Context**: Uses `models/model.h5` and `models/scaler.pkl`.

### D. Deployment
*   **Triggers**: Manual agent mission via `/gcp`.
*   **Processor**: `src/vertex_trigger.py`.
*   **Receiver**: Vertex AI Custom Jobs.
