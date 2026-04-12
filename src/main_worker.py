import os
import sys
# Path resolution for industrial architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, BackgroundTasks, HTTPException
from datetime import datetime
import uvicorn

# Ensure src is in python path
sys.path.append(os.path.dirname(__file__))

import data_loader
import vertex_trigger
from src.facades.forecasting import ForecastingFacade
from src.facades.lifecycle_facade import LifecycleFacade
from src.repositories.asset_repo import AssetRepository
from src.utils.logger import setup_logger

logger = setup_logger("worker.main")
forecaster = ForecastingFacade()
lifecycle_manager = LifecycleFacade()
assets = AssetRepository()

app = FastAPI(title="BTC Predictor Tactical Worker")

@app.get("/health")
def health():
    return {"status": "operational", "timestamp": datetime.now()}

@app.post("/recalibrate")
async def recalibrate():
    """
    Hourly Recalibration Task:
    1. Fetches latest market data.
    2. Runs model forecast with SYSTEM attribution.
    3. Logs results to the dedicated live_drift_audit table.
    """
    try:
        # 1. Fetch Fresh Market Data and Assets
        full_df, clean_df = data_loader.prepare_merged_dataset()
        
        # Ensure local artifacts are synchronized from GCS
        model_filename = "btc_lstm_model.h5"
        scaler_filename = "scaler.pkl"
        
        if not os.path.exists(os.path.join("models", model_filename)):
            logger.info(f"Model {model_filename} missing locally. Pulling from cloud...")
            assets.sync_from_cloud(model_filename)
            
        if not os.path.exists(os.path.join("models", scaler_filename)):
            logger.info(f"Scaler {scaler_filename} missing locally. Pulling from cloud...")
            assets.sync_from_cloud(scaler_filename)
            
        model = assets.load_model(model_filename)
        scaler = assets.load_scaler(scaler_filename)
        
        # 2. Trigger Headless Recalibration (SYSTEM source)
        results = forecaster.get_forecast(
            model=model, 
            scaler=scaler, 
            clean_df=clean_df,
            force=True,
            source="SYSTEM_AUTO"
        )
        
        return {
            "status": "success",
            "trigger": "SYSTEM_AUTO",
            "prediction": results['prices'][0],
            "drift_pct": results.get('avg_drift', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    """
    Daily Retraining Task:
    Triggers a new Vertex AI CustomJob for model refinement.
    """
    try:
        # Retrieve SA from environment (injected during Cloud Run deploy)
        res = lifecycle_manager.launch_retraining()
        return {
            "status": "training_launched",
            "job_id": res['job_id']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
