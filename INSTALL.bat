@echo off
REM One-time install: Python venv (.venv) + PyTorch + requirements, then frontend dependencies (if Node.js is present).
setlocal
cd /d "%~dp0"
call "%~dp0SETUP.bat" nopause
if exist "frontend\package.json" (
    where npm >nul 2>&1 && (
        pushd frontend
        if not exist ".env.local" copy ".env.local.example" ".env.local" >nul
        call npm ci
        popd
    ) || echo Node.js not found - skipping the Next.js frontend. The built-in web UI at http://127.0.0.1:8000 still works.
)
echo.
echo Install finished. Next: RUN.bat
if /i not "%~1"=="nopause" pause
