import os
import sys
# Path resolution for industrial architecture
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(root_path, "src")
sys.path.append(root_path)
sys.path.append(src_path)

from fastapi import FastAPI, BackgroundTasks, HTTPException
from datetime import datetime
import uvicorn

# Ensure src is in python path
sys.path.append(os.path.dirname(__file__))

from src.facades.forecasting import ForecastingFacade
from src.facades.lifecycle_facade import LifecycleFacade
from src.utils.logger import setup_logger

logger = setup_logger("worker.main")
forecaster = ForecastingFacade()
lifecycle_manager = LifecycleFacade()

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
        full_df = lifecycle_manager.load_dataset()
        model, scaler = lifecycle_manager.load_model_assets()
        clean_df = full_df # Standardized by orchestrator
        
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
        logger.exception(f"Fatal error during recalibration: {e}")
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
        logger.exception(f"Fatal error during retraining launch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
