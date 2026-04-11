$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = "$here\..\..\scripts\deploy_ui.ps1"

Describe "deploy_ui.ps1" {
    setup {
        Mock gcloud { return $true }
        Mock Write-Host { }
    }

    It "Sets the project context" {
        & $scriptPath
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'config' -and $args[1] -eq 'set' -and $args[2] -eq 'project'
        } -Exactly 1
    }

    It "Builds using the correct infra path" {
        & $scriptPath
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'builds' -and $args[1] -eq 'submit' -and $args[3] -eq 'infra/dashboard.yaml'
        } -Exactly 1
    }

    It "Deploys to Cloud Run with correct service name" {
        & $scriptPath
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'run' -and $args[1] -eq 'deploy' -and $args[2] -eq 'btc-dashboard'
        } -Exactly 1
    }
}
