import os
import sys
from fastapi import FastAPI, BackgroundTasks, HTTPException
from datetime import datetime
import uvicorn

# Ensure src is in python path
sys.path.append(os.path.dirname(__file__))

import data_loader
import dashboard_logic
import vertex_trigger
from database import DatabaseManager

app = FastAPI(title="BTC Predictor Tactical Worker")
db_mgr = DatabaseManager()

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
        # 1. Fetch Fresh Market Data
        full_df, clean_df = data_loader.prepare_merged_dataset()
        
        # 2. Trigger Headless Recalibration (SYSTEM source)
        results = dashboard_logic.get_base_forecast(clean_df, full_df, source="SYSTEM")
        
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
        sa_email = os.getenv("SERVICE_ACCOUNT")
        job = vertex_trigger.trigger_training_job(service_account=sa_email)
        return {
            "status": "training_launched",
            "job_name": job.display_name,
            "job_id": job.resource_name,
            "running_as": sa_email
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
