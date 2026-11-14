@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set TRANSFORMERS_VERBOSITY=error
call "%~dp0scripts\find_python.bat" || ( echo No Python environment. Run SETUP.bat. & pause & exit /b 1 )
"%PHYLLO_PY%" tests\run_tests.py
pause
