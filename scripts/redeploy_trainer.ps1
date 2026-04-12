# Targeted Deployment: BTC Trainer Only
$PROJECT_ID = "btc-predictor-492515"

Write-Host "1. Rebuilding BTC Trainer Image..." -ForegroundColor Cyan
gcloud builds submit --config infra/train.yaml . --project $PROJECT_ID

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n2. Trainer image updated. Launching Vertex AI Job..." -ForegroundColor Green
    
    # Use venv python if available to ensure dependencies (setuptools) are loaded
    $PythonPath = "python"
    if (Test-Path ".\venv\Scripts\python.exe") { 
        $PythonPath = ".\venv\Scripts\python.exe" 
        Write-Host "Using Virtual Environment: $PythonPath" -ForegroundColor Gray
    }
    
    & $PythonPath src/train_on_gcp.py
} else {
    Write-Error "Build failed. Training job not triggered."
}
