param([Parameter(ValueFromRemainingArguments=$true)]$RemainingArgs)

$logPath = Join-Path $PSScriptRoot "..\test_log.txt"
$commandline = "gcloud " + ($RemainingArgs -join " ")
Add-Content -Path $logPath -Value $commandline

# Simulate behaviors for specific commands if needed
if ($RemainingArgs[0] -eq "run" -and $RemainingArgs[1] -eq "services" -and $RemainingArgs[2] -eq "describe") {
    Write-Output "https://mock-worker-url.a.run.app"
}

if ($RemainingArgs[0] -eq "iam" -and $RemainingArgs[1] -eq "service-accounts" -and $RemainingArgs[2] -eq "list") {
    Write-Output "btc-forecaster-sa@btc-predictor-492515.iam.gserviceaccount.com"
}
