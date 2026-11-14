@echo off
REM Re-runs every experiment (E1-E6) from scratch and regenerates results/RESULTS.md and artifacts/.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set TRANSFORMERS_VERBOSITY=error
call "%~dp0scripts\find_python.bat" || (call "%~dp0SETUP.bat" nopause & call "%~dp0scripts\find_python.bat")
if not defined PHYLLO_PY ( echo No Python environment. Run SETUP.bat. & pause & exit /b 1 )
"%PHYLLO_PY%" experiments\run_all.py
echo.
echo Done. Open results\RESULTS.md for all tables.
pause
