<#
.SYNOPSIS
    Automated Deployment Orchestration for BTC Prediction Dashboard.
    
.DESCRIPTION
    1. Syncs local model weights to GCS.
    2. Submits Training and Dashboard builds to Google Cloud Build.
    3. Deploys the latest Dashboard image to Google Cloud Run.
#>

$PROJECT_ID = "btc-predictor-492515"
$REGION = "us-central1"
$BUCKET = "btc-predictor-492515_cloudbuild"
$DASHBOARD_IMAGE = "gcr.io/$PROJECT_ID/btc-dashboard"
$TRAINER_IMAGE = "gcr.io/$PROJECT_ID/btc-trainer"

Write-Host "STARTING BTC Predictor Deployment Pipeline..." -ForegroundColor Cyan

# 1. Synchronize Model Weights
Write-Host "`n[1/4] Syncing local weights to GCS..." -ForegroundColor Yellow
if (Test-Path "models/btc_lstm_model.h5") {
    gcloud storage cp models/btc_lstm_model.h5 "gs://$BUCKET/models/btc_lstm_model.h5"
    gcloud storage cp models/scaler.pkl "gs://$BUCKET/models/scaler.pkl"
    Write-Host "SUCCESS: Model weights synced." -ForegroundColor Green
} else {
    Write-Host "WARNING: local models/btc_lstm_model.h5 not found. Skipping weight sync." -ForegroundColor Red
}

# 2. Build Training Pipeline
Write-Host "`n[2/4] Submitting Training Pipeline build..." -ForegroundColor Yellow
gcloud builds submit --config infra/train.yaml . --project $PROJECT_ID
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed for Trainer."; exit $LASTEXITCODE }

# 3. Build Dashboard Application
Write-Host "`n[3/4] Submitting Dashboard App build..." -ForegroundColor Yellow
gcloud builds submit --config infra/app.yaml . --project $PROJECT_ID
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed for Dashboard."; exit $LASTEXITCODE }

# 4. Deploy to Cloud Run
Write-Host "`n[4/4] Deploying Dashboard to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy btc-dashboard `
    --image $DASHBOARD_IMAGE `
    --region $REGION `
    --project $PROJECT_ID `
    --memory 2Gi `
    --allow-unauthenticated

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nCOMPLETE: Pipeline Complete! BTC Predictor is live." -ForegroundColor Cyan
} else {
    Write-Error "Deployment failed."
}
