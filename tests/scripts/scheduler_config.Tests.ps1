$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = "$here\..\..\scripts\scheduler_config.ps1"

Describe "scheduler_config.ps1" {
    setup {
        Mock gcloud { 
            if ($args[0] -eq 'run' -and $args[1] -eq 'services' -and $args[5] -eq 'value(status.url)') {
                return "https://mock-worker.a.run.app"
            }
            # Simulate failure on first create call to trigger update logic
            if ($args[0] -eq 'scheduler' -and $args[1] -eq 'jobs' -and $args[2] -eq 'create') {
                $global:LASTEXITCODE = 1
                return $false
            }
            return $true 
        }
        Mock Write-Host { }
    }

    It "Attempts to create jobs and falls back to update on failure" {
        & $scriptPath

        # Should have called create twice (failed) and update twice (success)
        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'scheduler' -and $args[1] -eq 'jobs' -and $args[2] -eq 'create'
        } -Exactly 2

        Assert-MockCalled gcloud -ParameterFilter {
            $args[0] -eq 'scheduler' -and $args[1] -eq 'jobs' -and $args[2] -eq 'update'
        } -Exactly 2
    }
}
