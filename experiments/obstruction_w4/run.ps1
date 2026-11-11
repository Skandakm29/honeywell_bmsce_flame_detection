# Usage: .\run.ps1 <step>      e.g.  .\run.ps1 0      .\run.ps1 2
# Reuses the V10 venv, V10 code and V12's data11.py (read-only) and the V10 work folder.
# Week 4 writes only: manifests_w4, train_w4, select_w4, status_w4, outbox\w4_results.zip
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
if (-not $env:V10_PACKAGE) { $env:V10_PACKAGE = 'D:\Honeywell-V10' }
if (-not $env:V12_PACKAGE) { $env:V12_PACKAGE = 'D:\Honeywell-V12' }
if (-not $env:V10_WORK) { $env:V10_WORK = 'D:\Honeywell-V10-work' }
$env:PYTHONUNBUFFERED = '1'
$python = Join-Path $env:V10_PACKAGE '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "V10 venv not found at $python (set `$env:V10_PACKAGE)" }
$log = Join-Path $env:V10_WORK ('w4_step_' + $args[0] + '.log')
Write-Host "V10 package: $env:V10_PACKAGE   V12 package: $env:V12_PACKAGE   Work folder: $env:V10_WORK   Log: $log"
& $python -u (Join-Path $root 'workflow_w4\run_w4.py') @args 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Write-Host "`nW4 STEP FAILED. See $log. Rerunning the same command resumes." -ForegroundColor Red; exit 1 }
