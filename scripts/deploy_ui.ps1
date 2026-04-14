# Configuration
$PROJECT_ID = "btc-predictor-492515"
$SERVICE_NAME = "btc-dashboard"
$REGION = "us-central1"

Write-Host "Step 0: Auto-Focusing Project Context ($PROJECT_ID)..." -ForegroundColor Cyan
gcloud config set project $PROJECT_ID

Write-Host "Step 1: Building and Deploying Dashboard (Industrial Profile)..." -ForegroundColor Cyan

# A. Build the dashboard image using the specific YAML config
gcloud builds submit --config infra/dashboard.yaml .

# B. Deploy the dashboard image
gcloud run deploy $SERVICE_NAME `
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME `
    --region $REGION `
    --platform managed `
    --set-env-vars 'DASHBOARD_PASSWORD=btc1984,DASHBOARD_PASS_1=E(nC>D8<<279oNV@,DASHBOARD_PASS_2=727C?O%yDb-G?4=N,DASHBOARD_PASS_3=1(OcJ=1j$uE"p98I,DASHBOARD_PASS_4=%k):7J5j5MY<f/_O' `
    --quiet

Write-Host ""
Write-Host "Dashboard Restore Complete. Visit your URL to see the Tactical HUD!" -ForegroundColor Green
