# Configuration
$ProjectId = "btc-predictor-492515"
$ServiceName = "btc-tactical-worker"
$Region = "us-central1"
$SaName = "btc-forecaster-sa"
$SaEmail = "$SaName@$ProjectId.iam.gserviceaccount.com"

Write-Host "Step 0: Auto-Focusing Project Context ($ProjectId)..." -ForegroundColor Cyan
gcloud config set project $ProjectId

Write-Host "Step 1: Enabling Cloud Scheduler API (Verification)..." -ForegroundColor Cyan
gcloud services enable cloudscheduler.googleapis.com --quiet

Write-Host "Step 2: Retrieving live Worker URL..." -ForegroundColor Cyan
$WorkerUrl = gcloud run services describe $ServiceName --platform managed --region $Region --format 'value(status.url)'
if (!$WorkerUrl) {
    Write-Host "ERROR: Could not find service $ServiceName. Please ensure Phase 3 deployment passed." -ForegroundColor Red
    exit 1
}
Write-Host "Target URL: $WorkerUrl" -ForegroundColor Green

Write-Host "Step 3: Configuring Secure Cloud Scheduler (Authenticated)..." -ForegroundColor Cyan

# 1. Hourly Recalibration
Write-Host "Creating/Updating Hourly Recalibration Schedule..."
gcloud scheduler jobs create http btc-hourly-recalibrate `
    --schedule="0 * * * *" `
    --uri="$WorkerUrl/recalibrate" `
    --http-method=POST `
    --location=$Region `
    --oidc-service-account-email=$SaEmail `
    --oidc-token-audience=$WorkerUrl `
    --quiet 2>$null

if ($LASTEXITCODE -ne 0) {
    # If job exists, update instead
    gcloud scheduler jobs update http btc-hourly-recalibrate `
        --schedule="0 * * * *" `
        --uri="$WorkerUrl/recalibrate" `
        --location=$Region `
        --oidc-service-account-email=$SaEmail `
        --oidc-token-audience=$WorkerUrl `
        --quiet
}

# 2. Daily Retraining
Write-Host "Creating/Updating Daily Retraining Schedule..."
gcloud scheduler jobs create http btc-daily-retrain `
    --schedule="0 0 * * *" `
    --uri="$WorkerUrl/retrain" `
    --http-method=POST `
    --location=$Region `
    --oidc-service-account-email=$SaEmail `
    --oidc-token-audience=$WorkerUrl `
    --quiet 2>$null

if ($LASTEXITCODE -ne 0) {
    # If job exists, update instead
    gcloud scheduler jobs update http btc-daily-retrain `
        --schedule="0 0 * * *" `
        --uri="$WorkerUrl/retrain" `
        --location=$Region `
        --oidc-service-account-email=$SaEmail `
        --oidc-token-audience=$WorkerUrl `
        --quiet
}

Write-Host ""
Write-Host "Automation Activated. Your cycles are now live." -ForegroundColor Green
