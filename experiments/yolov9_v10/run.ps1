# Usage: .\run.ps1 <step> [options]     e.g.  .\run.ps1 1 --clips E:\V10_clips --fasdd-zip E:\dataset.zip
# Work data (datasets, checkpoints, outbox) lives OUTSIDE the git clone, in $env:V10_WORK (default: ..\Honeywell-V10-work).
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
if (-not $env:V10_WORK) { $env:V10_WORK = Join-Path (Split-Path $root -Parent) 'Honeywell-V10-work' }
New-Item -ItemType Directory -Force -Path $env:V10_WORK | Out-Null
$env:HF_HOME = Join-Path $env:V10_WORK 'hf'
$env:KAGGLEHUB_CACHE = Join-Path $env:V10_WORK 'kaggle-cache'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:PYTHONUNBUFFERED = '1'
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run .\setup.ps1 first.' }
$log = Join-Path $env:V10_WORK ('step_' + $args[0] + '.log')
Write-Host "Work folder: $env:V10_WORK   Log: $log"
& $python -u (Join-Path $root 'workflow\run_v10.py') @args 2>&1 | Tee-Object -FilePath $log -Append
if ($LASTEXITCODE -ne 0) { Write-Host "`nSTEP FAILED. Send $log (last 60 lines are enough). Rerunning the same command resumes." -ForegroundColor Red; exit 1 }
