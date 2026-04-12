# Targeted Deployment: BTC Trainer Only
$PROJECT_ID = "btc-predictor-492515"

Write-Host "1. Rebuilding BTC Trainer Image..." -ForegroundColor Cyan
gcloud builds submit --config infra/train.yaml . --project $PROJECT_ID

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n2. Trainer image updated. Launching Vertex AI Job..." -ForegroundColor Green
    
    # Priority: Use ambient 'python' if we are in a validation/shadowing environment
    $PythonPath = "python"
    if ((Test-Path ".\venv\Scripts\python.exe") -and !($env:PATH -like "*tests\bin*")) { 
        $PythonPath = ".\venv\Scripts\python.exe" 
        Write-Host "Using Virtual Environment: $PythonPath" -ForegroundColor Gray
    }
    
    & $PythonPath src/vertex_trigger.py
} else {
    Write-Error "Build failed. Training job not triggered."
}
