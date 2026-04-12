param([Parameter(ValueFromRemainingArguments=$true)]$RemainingArgs)

$logPath = Join-Path $PSScriptRoot "..\test_log.txt"
$commandline = "python " + ($RemainingArgs -join " ")
Add-Content -Path $logPath -Value $commandline

# Handle specific mock behaviors if needed
if ($RemainingArgs -like "*src/vertex_trigger.py*") {
    Write-Host "Shadow Trace: Simulation of Vertex AI Job Launch successful." -ForegroundColor DarkGray
}
exit 0
