@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0lab_setup.ps1"
if errorlevel 1 echo Training did not finish. Please send the last 30 lines of run.log or setup.log.
pause
