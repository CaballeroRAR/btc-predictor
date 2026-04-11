$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = "$here\..\..\scripts\deploy_all.ps1"

Describe "deploy_all.ps1" {
    setup {
        # Define functions to shadow native commands
        function gcloud { }
        function gsutil { }
        
        Mock gcloud { return $true }
        Mock gsutil { return $true }
        Mock Write-Host { }
        Mock Test-Path { return $true }
        Mock Write-Error { }
    }

    It "Calls gcloud storage cp for model weights" {
        # Execute script
        & $scriptPath

        # Verify gcloud storage cp was called
        Assert-MockCalled gcloud -ParameterFilter { 
            $args[0] -eq 'storage' -and $args[1] -eq 'cp' -and $args[2] -like 'models/*'
        } -Exactly 2
    }

    It "Submits builds for both Trainer and Dashboard" {
        & $scriptPath

        # Verify Trainer build
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'builds' -and $args[1] -eq 'submit' -and $args[3] -eq 'infra/train.yaml'
        } -Exactly 1

        # Verify Dashboard build
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'builds' -and $args[1] -eq 'submit' -and $args[3] -eq 'infra/dashboard.yaml'
        } -Exactly 1
    }

    It "Deploys the Dashboard to Cloud Run" {
        & $scriptPath

        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'run' -and $args[1] -eq 'deploy' -and $args[2] -eq 'btc-dashboard'
        } -Exactly 1
    }
}
