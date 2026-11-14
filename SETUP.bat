@echo off
REM ==========================================================================
REM  First-time setup: creates .venv and installs PyTorch (CUDA if possible) +
REM  the pinned requirements. Only needed on a machine without a ready env.
REM ==========================================================================
setlocal
cd /d "%~dp0"
set "VPY=%~dp0.venv\Scripts\python.exe"

if not exist "%VPY%" (
    echo Creating virtual environment in .venv ...
    py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
)
if not exist "%VPY%" (
    echo Could not create a virtual environment. Install Python 3.12 from python.org and retry.
    goto :end
)
"%VPY%" -m pip install --upgrade pip
echo.
echo Installing PyTorch with CUDA support (falls back to CPU build if unavailable)...
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 || "%VPY%" -m pip install torch torchvision
echo.
echo Installing project requirements...
"%VPY%" -m pip install -r requirements.txt
"%VPY%" -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo Setup finished.
:end
if /i not "%~1"=="nopause" pause
