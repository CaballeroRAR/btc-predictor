$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = "$here\..\..\scripts\worker_init.ps1"

Describe "worker_init.ps1" {
    setup {
        # Mock gcloud and other utilities
        Mock gcloud { 
            if ($args[0] -eq 'run' -and $args[1] -eq 'services' -and $args[5] -eq 'value(status.url)') {
                return "https://mock-worker.a.run.app"
            }
            return $true 
        }
        Mock gsutil { return $true }
        Mock Write-Host { }
        Mock gcloud { return "btc-forecaster-sa@btc-predictor-492515.iam.gserviceaccount.com" } -Tag "sa_check" -ParameterFilter { $args[0] -eq "iam" -and $args[1] -eq "service-accounts" }
    }

    It "Checks for service account existence" {
        & $scriptPath
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'iam' -and $args[1] -eq 'service-accounts' -and $args[2] -eq 'list'
        } -Exactly 1
    }

    It "Provisions storage bucket if missing" {
        & $scriptPath
        Assert-MockCalled gsutil -ParameterFilter {
            $args[0] -eq 'ls' -and $args[1] -eq '-b'
        } -AtLeast 1
    }

    It "Builds using the correct infra path" {
        & $scriptPath
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'builds' -and $args[1] -eq 'submit' -and $args[3] -eq 'infra/worker.yaml'
        } -Exactly 1
    }

    It "Deploys the tactical worker with correct variables" {
        & $scriptPath
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'run' -and $args[1] -eq 'deploy' -and $args[2] -eq 'btc-tactical-worker' -and $args[16] -like '*PROJECT_ID=*'
        } -Exactly 1
    }

    It "Configures scheduler jobs for recalibration and retraining" {
        & $scriptPath
        # Recalibrate Job
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'scheduler' -and $args[1] -eq 'jobs' -and $args[3] -eq 'btc-hourly-recalibrate'
        } -Exactly 1
        
        # Retrain Job
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'scheduler' -and $args[1] -eq 'jobs' -and $args[3] -eq 'btc-daily-retrain'
        } -Exactly 1
    }
}
