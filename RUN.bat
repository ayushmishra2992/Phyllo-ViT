@echo off
REM Starts the backend (http://127.0.0.1:8000, includes a built-in UI) and, if Node.js is installed,
REM the Next.js frontend (http://localhost:3000/lab) in a second window.
setlocal
cd /d "%~dp0"
where npm >nul 2>&1 && if exist "frontend\node_modules" start "Phyllo-ViT frontend" cmd /c "%~dp0START_FRONTEND.bat"
call "%~dp0RUN_ME.bat"
