@echo off
REM ==========================================================================
REM  Phyllo-ViT v3 - one-click launcher
REM  Finds a working Python environment, builds artifacts if missing,
REM  starts the API + web UI and opens http://127.0.0.1:8000 in your browser.
REM ==========================================================================
setlocal
cd /d "%~dp0"
title Phyllo-ViT v3
set PYTHONIOENCODING=utf-8
set TRANSFORMERS_VERBOSITY=error

call "%~dp0scripts\find_python.bat"
if errorlevel 1 (
    echo.
    echo No ready Python environment found. Running first-time setup ^(downloads ~3 GB, 10-20 min^)...
    call "%~dp0SETUP.bat" nopause
    call "%~dp0scripts\find_python.bat"
    if errorlevel 1 goto :fail
)
echo Using Python: %PHYLLO_PY%

if not exist "artifacts\reliability_ft4.json" (
    echo Reliability artifacts missing - running experiments once ^(about 6-10 minutes on GPU^)...
    "%PHYLLO_PY%" experiments\run_all.py || goto :fail
)

echo.
echo Starting Phyllo-ViT. The browser opens automatically when the models are loaded.
echo Keep this window open. Press Ctrl+C to stop.
echo   Web UI : http://127.0.0.1:8000
echo   API docs: http://127.0.0.1:8000/docs
echo.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 180;$i++){try{$r=Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 2; if($r.status -eq 'healthy'){Start-Process 'http://127.0.0.1:8000'; break}}catch{}; Start-Sleep 1}"
"%PHYLLO_PY%" -m uvicorn api.main:app --host 127.0.0.1 --port 8000
goto :eof

:fail
echo.
echo Something went wrong. See the messages above, or README.md - Troubleshooting.
pause
exit /b 1
