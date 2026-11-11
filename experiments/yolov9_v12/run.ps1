# Usage: .\run.ps1 <step>      e.g.  .\run.ps1 0      .\run.ps1 4a
# Reuses the V10 venv, V10 code (read-only), the V10 work folder and V11's manifests_v11 (read-only).
# V12 writes only: *_v12 folders, bench\v12*, temporal\v12*, diagnose\v12, status_v12, REPORT_V12.md, outbox\v12_*.zip
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
if (-not $env:V10_PACKAGE) { $env:V10_PACKAGE = 'D:\Honeywell-V10' }
if (-not $env:V10_WORK) { $env:V10_WORK = 'D:\Honeywell-V10-work' }
$env:HF_HOME = Join-Path $env:V10_WORK 'hf'
$env:KAGGLEHUB_CACHE = Join-Path $env:V10_WORK 'kaggle-cache'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUNBUFFERED = '1'
$python = Join-Path $env:V10_PACKAGE '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw "V10 venv not found at $python (set `$env:V10_PACKAGE)" }
$log = Join-Path $env:V10_WORK ('v12_step_' + $args[0] + '.log')
Write-Host "V10 package: $env:V10_PACKAGE   Work folder: $env:V10_WORK   Log: $log"
& $python -u (Join-Path $root 'workflow12\run_v12.py') @args 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Write-Host "`nV12 STEP FAILED. See $log. Rerunning the same command resumes." -ForegroundColor Red; exit 1 }
