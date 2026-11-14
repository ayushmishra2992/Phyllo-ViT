@echo off
REM Checks Alternate / Opposite / Whorled / Not_a_leaf on every held-out image and through HTTP /predict.
setlocal
cd /d "%~dp0"
call "%~dp0scripts\find_python.bat" || ( echo No Python environment found. Run SETUP.bat first. & pause & exit /b 1 )
"%PHYLLO_PY%" scripts\check_classes.py
pause
