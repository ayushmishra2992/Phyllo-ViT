@echo off
REM Sets PHYLLO_PY to the first Python that has every required package. Exit code 1 if none.
set "PHYLLO_PY="
set "CHECK=import torch, transformers, fastapi, uvicorn, multipart, sklearn, scipy, pandas, openpyxl, matplotlib, skimage, joblib, httpx"
for %%P in (
    "%~dp0..\.venv\Scripts\python.exe"
    "%USERPROFILE%\OneDrive\Desktop\Phyllotaxy\.venv\Scripts\python.exe"
    "%USERPROFILE%\Desktop\Phyllotaxy\.venv\Scripts\python.exe"
    "%~dp0..\..\Phyllotaxy\.venv\Scripts\python.exe"
) do (
    if exist %%P (
        %%P -c "%CHECK%" >nul 2>&1 && (
            set "PHYLLO_PY=%%~fP"
            exit /b 0
        )
    )
)
exit /b 1
