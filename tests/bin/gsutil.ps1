param([Parameter(ValueFromRemainingArguments=$true)]$RemainingArgs)

$logPath = Join-Path $PSScriptRoot "..\test_log.txt"
$commandline = "gsutil " + ($RemainingArgs -join " ")
Add-Content -Path $logPath -Value $commandline
