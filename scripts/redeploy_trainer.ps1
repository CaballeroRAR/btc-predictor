# Targeted Deployment: BTC Trainer Only
$ProjectId = "btc-predictor-492515"

Write-Host "1. Rebuilding BTC Trainer Image (14-Feature Harden)..." -ForegroundColor Cyan
gcloud builds submit --config infra/train.yaml . --project $ProjectId

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n2. Trainer image updated. Launching Vertex AI Job..." -ForegroundColor Green
    python src/train_on_gcp.py
} else {
    Write-Error "Build failed. Training job not triggered."
}
