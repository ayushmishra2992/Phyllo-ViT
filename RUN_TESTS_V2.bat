@echo off
REM Runs the v1 suite (13 tests) and the v2 suite.
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set TRANSFORMERS_VERBOSITY=error
call "%~dp0scripts\find_python.bat" || ( echo No Python environment. Run INSTALL.bat. & pause & exit /b 1 )
"%PHYLLO_PY%" tests\run_tests.py || goto :done
"%PHYLLO_PY%" tests\run_tests_v2.py
:done
pause
