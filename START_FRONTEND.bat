@echo off
REM Optional: Riya's Next.js frontend (needs Node.js 20.9+). The backend (RUN_ME.bat) must be running.
setlocal
cd /d "%~dp0frontend"
where npm >nul 2>&1 || ( echo Node.js/npm not found. Install Node.js LTS from nodejs.org. & pause & exit /b 1 )
if not exist ".env.local" copy ".env.local.example" ".env.local" >nul
if not exist "node_modules" ( echo Installing frontend dependencies... & call npm install )
echo Frontend: http://localhost:3000   (backend must be running on http://127.0.0.1:8000)
start "" cmd /c "timeout /t 8 >nul & start http://localhost:3000/lab"
call npm run dev
