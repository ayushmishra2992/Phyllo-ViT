@echo off
REM Phyllo-ViT 2.0: backend (v1 + /v2 routes) on http://127.0.0.1:8000 and, if Node.js is installed, the Next.js UI.
REM Set OPENAQ_API_KEY first if you want ambient air-quality lookups (free key: https://explore.openaq.org/register).
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set TRANSFORMERS_VERBOSITY=error
call "%~dp0scripts\find_python.bat"
if errorlevel 1 ( echo No Python environment found. Run INSTALL.bat first. & pause & exit /b 1 )
echo Using Python: %PHYLLO_PY%
if not exist "artifacts\reliability_ft4.json" ( echo Missing artifacts - run RUN_EXPERIMENTS.bat once. & pause & exit /b 1 )
where npm >nul 2>&1 && if exist "frontend\node_modules" start "Phyllo-ViT 2.0 frontend" cmd /c "cd /d %~dp0frontend && (if not exist .env.local copy .env.local.example .env.local >nul) && start http://localhost:3000/v2 && npm run dev"
echo   2.0 UI (needs Node): http://localhost:3000/v2     v1 UI: http://127.0.0.1:8000     API docs: http://127.0.0.1:8000/docs
"%PHYLLO_PY%" -m uvicorn api.v2_app:app --host 127.0.0.1 --port 8000
