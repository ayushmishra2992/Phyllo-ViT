@echo off
REM Builds ..\Phyllo-ViT-Final-Ready.zip (no .venv, node_modules, caches or secrets).
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\package_final.ps1"
pause
