# Temporal alarm rule for V10 + small-box threshold. Reads V10/V12 logs, writes only <work>\temporal_final\.
$ErrorActionPreference = 'Stop'
if (-not $env:V10_PACKAGE) { $env:V10_PACKAGE = 'D:\Honeywell-V10' }
if (-not $env:V10_WORK) { $env:V10_WORK = 'D:\Honeywell-V10-work' }
$env:PYTHONPATH = Join-Path $env:V10_PACKAGE 'workflow'
$env:PYTHONUNBUFFERED = '1'
$python = Join-Path $env:V10_PACKAGE '.venv\Scripts\python.exe'
$log = Join-Path $env:V10_WORK 'temporal_final.log'
& $python -u (Join-Path $PSScriptRoot 'temporal_final.py') 2>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED, see $log" -ForegroundColor Red; exit 1 }
$zip = Join-Path $env:V10_WORK 'outbox\temporal_final_results.zip'
$out = Join-Path $env:V10_WORK 'temporal_final'
Compress-Archive -Force -Path (Join-Path $out '*.json'), (Join-Path $out '*.md'), $log -DestinationPath $zip
Write-Host "Done: $zip"
