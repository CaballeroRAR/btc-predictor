<#
.SYNOPSIS
    Automated Deployment Orchestration for BTC Prediction Dashboard.
    
.DESCRIPTION
    1. Syncs local model weights to GCS.
    2. Submits Training and Dashboard builds to Google Cloud Build.
    3. Deploys the latest Dashboard image to Google Cloud Run.
#>

# --- Environment & Secret Loading ---
$DOTENV_PATH = ".env"
if (Test-Path $DOTENV_PATH) {
    Write-Host "Loading environment from $DOTENV_PATH..." -ForegroundColor Gray
    Get-Content $DOTENV_PATH | Where-Object { $_ -match '=' -and $_ -notmatch '^#' } | ForEach-Object {
        $name, $value = $_ -split '=', 2
        Set-Variable -Name $name.Trim() -Value $value.Trim() -Scope Global
    }
} else {
    Write-Error "CRITICAL: .env file missing. Cannot proceed with deployment."
    exit 1
}

# Calculated Globals
$DASHBOARD_IMAGE = "gcr.io/$PROJECT_ID/btc-dashboard"
$TRAINER_IMAGE = "gcr.io/$PROJECT_ID/btc-trainer"

Write-Host "STARTING BTC Predictor Deployment Pipeline..." -ForegroundColor Cyan

# 1. Synchronize Model Weights
Write-Host "`n[1/4] Syncing local weights to GCS..." -ForegroundColor Yellow
if (Test-Path "models/btc_lstm_model.h5") {
    gcloud storage cp models/btc_lstm_model.h5 "gs://$BUCKET_NAME/models/btc_lstm_model.h5"
    gcloud storage cp models/scaler.pkl "gs://$BUCKET_NAME/models/scaler.pkl"
    Write-Host "SUCCESS: Model weights synced." -ForegroundColor Green
} else {
    Write-Host "WARNING: local models/btc_lstm_model.h5 not found. Skipping weight sync." -ForegroundColor Red
}

# 2. Build Training Pipeline
Write-Host "`n[2/4] Submitting Training Pipeline build..." -ForegroundColor Yellow
gcloud builds submit --config infra/train.yaml . --project $PROJECT_ID
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed for Trainer."; exit $LASTEXITCODE }

# 2.5 Build Tactical Worker
Write-Host "`n[2.5/4] Submitting Tactical Worker build..." -ForegroundColor Yellow
gcloud builds submit --config infra/worker.yaml . --project $PROJECT_ID
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed for Worker."; exit $LASTEXITCODE }

# 3. Build Dashboard Application
Write-Host "`n[3/4] Submitting Dashboard App build..." -ForegroundColor Yellow
gcloud builds submit --config infra/dashboard.yaml . --project $PROJECT_ID
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed for Dashboard."; exit $LASTEXITCODE }

# 4. Deploy to Cloud Run
Write-Host "`n[4/4] Deploying Dashboards and Workers to Cloud Run..." -ForegroundColor Yellow

$SA_EMAIL = "btc-forecaster-sa@$PROJECT_ID.iam.gserviceaccount.com"
$PASSWORDS = "DASHBOARD_PASSWORD=$DASHBOARD_PASSWORD,DASHBOARD_PASS_1=$DASHBOARD_PASS_1,DASHBOARD_PASS_2=$DASHBOARD_PASS_2,DASHBOARD_PASS_3=$DASHBOARD_PASS_3,DASHBOARD_PASS_4=$DASHBOARD_PASS_4"
$ENV_VARS = "PROJECT_ID=$PROJECT_ID,SERVICE_ACCOUNT=$SA_EMAIL,BUCKET_NAME=$BUCKET_NAME,FIRESTORE_DATABASE=btc-pred-db,$PASSWORDS"

# 4a. Dashboard Deploy
gcloud run deploy btc-dashboard `
    --image $DASHBOARD_IMAGE `
    --region $REGION `
    --project $PROJECT_ID `
    --memory 2Gi `
    --set-env-vars $ENV_VARS `
    --allow-unauthenticated

# 4b. Tactical Worker Deploy
gcloud run deploy btc-tactical-worker `
    --image "gcr.io/$PROJECT_ID/btc-tactical-worker" `
    --region $REGION `
    --project $PROJECT_ID `
    --memory 2Gi `
    --service-account $SA_EMAIL `
    --set-env-vars $ENV_VARS `
    --no-allow-unauthenticated

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nCOMPLETE: Pipeline Complete! All BTC Predictor services are live." -ForegroundColor Cyan
} else {
    Write-Error "Deployment failed."
}
