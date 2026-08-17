@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Development environment not found.
    echo Run setup_windows_cpu.ps1 or setup_windows_cuda.ps1 first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" homr_gui.py
if errorlevel 1 pause
