# BTC Predictor Project

This system predicts Bitcoin prices and sentiment drift using a stacked LSTM model. It features a training pipeline (local/Vertex AI) and an interactive Streamlit dashboard (Cloud Run).

## Architectural Principles & Solved Quirks

Our infrastructure has stabilized over standard GCP setups by solving these complex issues natively:
1. **Keras Deserialization Mismatch**: Keras 3 often embeds micro-version specific metadata (like `quantization_config`). We completely eliminated these bugs by:
    - Strictly pinning `keras==3.12.1` across all environments.
    - Exporting models in the stable HDF5 format (`btc_lstm_model.h5`).
    - Using `compile=False` during loading for smooth inference.
2. **PyTrends Rate Limits**: Cloud Run IP addresses often hit Google Trends' HTTP 429 errors. We implemented a fallback fallback system: if search trends are blocked, the system defaults to a baseline proxy value (`50.0`) instead of crashing the dashboard.

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
    - **Bucket**: `btc-predictor-492515_cloudbuild`

---

## Local Development Workflow

To work locally without deploying every change:

### 1. Train the Model (Optional)
Generates the `.h5` model and the `scaler.pkl` in the `models/` folder.
```powershell
python src/train.py
```

### 2. Upload to Cloud Storage
If you trained a new model locally, sync it so Cloud Run can fetch it later:
```powershell
gcloud storage cp models/btc_lstm_model.h5 gs://btc-predictor-492515_cloudbuild/models/btc_lstm_model.h5
gcloud storage cp models/scaler.pkl gs://btc-predictor-492515_cloudbuild/models/scaler.pkl
```

### 3. Run the Dashboard Locally
Test your UI and logic on your own machine.
```powershell
streamlit run src/drift_analysis.py
```

---

## Deployment Workflow

When you are ready to publish changes to GCP:

### 1. Build and Deploy Dashboard
To update the Streamlit app on Cloud Run:
```powershell
# 1. Build the image
gcloud builds submit --config cloudbuild.app.yaml .

# 2. Deploy to Cloud Run
gcloud run deploy btc-dashboard --image=gcr.io/btc-predictor-492515/btc-dashboard --region=us-central1 --project=btc-predictor-492515 --memory=2Gi
```

### 2. Build the Trainer (Future/Optional)
To update the Vertex AI training container for cloud training:
```powershell
gcloud builds submit --config cloudbuild.train.yaml .
```

---

## Project Structure

- `src/drift_analysis.py`: Main Streamlit Dashboard UI and inference logic.
- `src/data_loader.py`: Market data and Google Trends ingestion handling.
- `src/train.py`: Local training pipeline script.
- `src/model.py`: Neural Network architecture definitions.
- `src/cloud_config.py`: Centralized GCP config, paths, and hyperparameters.
- `src/calibration.py`: Sentimental drift adjustments.
- `src/investments_manager.py`: JSON-based local/GCS storage for journaling positions.
- `src/model_lifecycle.py`: Evaluation metrics for model staleness.
