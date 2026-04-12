# Validation Test Runner for Deployment Scripts
$ErrorActionPreference = "Stop"

$here = $PSScriptRoot
$logFile = Join-Path $here "test_log.txt"
$binDir = Join-Path $here "bin"
$scriptsDir = Join-Path $here "..\scripts"

# 1. Prepare Environment
Write-Host "Setting up validation environment..." -ForegroundColor Cyan
if (Test-Path $logFile) { Remove-Item $logFile }

# Save original path to restore later
$oldPath = $env:PATH
$env:PATH = "$binDir;" + $env:PATH

function Run-Validation {
    param($ScriptName)
    Write-Host "`nValidation RUN: $ScriptName" -ForegroundColor Yellow
    $fullPath = Join-Path $scriptsDir $ScriptName
    try {
        & $fullPath
    } catch {
        Write-Warning "Caught expected non-critical failure or script completion: $_"
    }
}

try {
    # 2. Execute scripts
    Run-Validation "deploy_all.ps1"
    Run-Validation "deploy_ui.ps1"
    Run-Validation "worker_init.ps1"
    Run-Validation "scheduler_config.ps1"
    Run-Validation "redeploy_trainer.ps1"

    # 3. Analyze Logs
    Write-Host "`n--- CAPTURED OPERATIONS LOG ---" -ForegroundColor Cyan
    if (Test-Path $logFile) {
        $logs = Get-Content $logFile
        $logs | ForEach-Object { Write-Host "  [SHADOWED] $_" -ForegroundColor DarkGray }

        Write-Host "`nVerifying Logic Integrity..." -ForegroundColor Cyan
        
        $checks = @(
            @{ Name="Worker Infra Path"; Pattern="gcloud builds submit --config infra/worker.yaml" },
            @{ Name="Dashboard Infra Path"; Pattern="gcloud builds submit --config infra/dashboard.yaml" },
            @{ Name="Trainer Infra Path"; Pattern="gcloud builds submit --config infra/train.yaml" },
            @{ Name="Cloud Run Service"; Pattern="gcloud run deploy btc-dashboard" },
            @{ Name="Worker Run Service"; Pattern="gcloud run deploy btc-tactical-worker" },
            @{ Name="Env Var Serialization"; Pattern="--set-env-vars.*PROJECT_ID=btc-predictor-492515" },
            @{ Name="Scheduler Recalibrate"; Pattern="--uri=.*?/recalibrate" },
            @{ Name="Trainer Entry Point"; Pattern="src/vertex_trigger.py" }
        )

        $allPassed = $true
        foreach ($check in $checks) {
            $match = $logs | Select-String -Pattern $check.Pattern
            if ($match) {
                Write-Host "[PASS] $($check.Name)" -ForegroundColor Green
            } else {
                Write-Host "[FAIL] $($check.Name) (Pattern: $($check.Pattern))" -ForegroundColor Red
                $allPassed = $false
            }
        }

        if ($allPassed) {
            Write-Host "`nSUCCESS: All deployment scripts are logically valid and synchronized." -ForegroundColor Green
        } else {
            Write-Error "Validation failed. Inconsistencies found in captured logs."
        }
    } else {
        Write-Error "No log file generated. Shadowing may have failed."
    }

} finally {
    # Restore environment
    $env:PATH = $oldPath
    Write-Host "`nValidation environment restored." -ForegroundColor Cyan
}
