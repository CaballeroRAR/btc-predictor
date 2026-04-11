# Configuration
$ProjectId = "btc-predictor-492515"
$ServiceName = "btc-dashboard"
$Region = "us-central1"

Write-Host "Step 0: Auto-Focusing Project Context ($ProjectId)..." -ForegroundColor Cyan
gcloud config set project $ProjectId

Write-Host "Step 1: Building and Deploying Dashboard (Industrial Profile)..." -ForegroundColor Cyan

# A. Build the dashboard image using the specific YAML config
gcloud builds submit --config infra/dashboard.yaml .

# B. Deploy the dashboard image
gcloud run deploy $ServiceName `
    --image gcr.io/$ProjectId/$ServiceName `
    --region $Region `
    --platform managed `
    --quiet

Write-Host ""
Write-Host "Dashboard Restore Complete. Visit your URL to see the Tactical HUD!" -ForegroundColor Green
