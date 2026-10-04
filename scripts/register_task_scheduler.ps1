<#
.SYNOPSIS
Registers the two Phase 0 capture jobs as Windows Task Scheduler tasks:
  - AlgoTrade-IntradayCapture: starts a few minutes before market open (09:10 IST),
    streams the live feed until market close.
  - AlgoTrade-EODCapture: runs shortly after close (15:45 IST), pulls/reconciles the
    day's daily bar.

.NOTES
Run this yourself, in an elevated ("Run as Administrator") PowerShell prompt, after:
  1. GROWW_API_KEY, GROWW_API_SECRET, and POSTGRES_PASSWORD are set as Windows
     system/user environment variables (see README.md). Task Scheduler reads env vars
     from the Windows environment at task-run time, not from your current shell.
  2. `docker compose up -d` has been run and `python scripts/apply_migrations.py`
     has applied the schema.
  3. `python scripts/check_env.py` passes.

This is intentionally NOT something this codebase runs for you automatically:
registering scheduled tasks -- and choosing "run whether logged on or not" (which
requires storing your Windows password with the task) -- is a decision you should make
deliberately on your own machine, not something a script does silently.
#>

param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$PythonExe = (Get-Command python).Source
)

$intradayAction = New-ScheduledTaskAction -Execute $PythonExe `
    -Argument "-m algotrade.capture_intraday" -WorkingDirectory $ProjectRoot
$eodAction = New-ScheduledTaskAction -Execute $PythonExe `
    -Argument "-m algotrade.capture_eod" -WorkingDirectory $ProjectRoot

$intradayTrigger = New-ScheduledTaskTrigger -Daily -At "09:10AM"
$eodTrigger = New-ScheduledTaskTrigger -Daily -At "03:45PM"

$settings = New-ScheduledTaskSettingsSet `
    -WakeToRun -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
    -DontStopOnIdleEnd

Write-Host "This will prompt for your Windows account password, so the tasks can run whether you're logged on or not."
$cred = Get-Credential -Message "Windows credentials for the AlgoTrade scheduled tasks" -UserName "$env:USERDOMAIN\$env:USERNAME"

Register-ScheduledTask -TaskName "AlgoTrade-IntradayCapture" -Action $intradayAction `
    -Trigger $intradayTrigger -Settings $settings `
    -User $cred.UserName -Password $cred.GetNetworkCredential().Password `
    -RunLevel Highest -Force | Out-Null

Register-ScheduledTask -TaskName "AlgoTrade-EODCapture" -Action $eodAction `
    -Trigger $eodTrigger -Settings $settings `
    -User $cred.UserName -Password $cred.GetNetworkCredential().Password `
    -RunLevel Highest -Force | Out-Null

Write-Host "Registered AlgoTrade-IntradayCapture (09:10 daily) and AlgoTrade-EODCapture (15:45 daily)."
Write-Host "Verify with: Get-ScheduledTask -TaskName 'AlgoTrade-*'"
Write-Host "Trigger a manual test run with: Start-ScheduledTask -TaskName 'AlgoTrade-EODCapture'"
