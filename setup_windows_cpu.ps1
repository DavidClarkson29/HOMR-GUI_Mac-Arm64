$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
}

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install -e .
& $Python -m pip install pytest pyinstaller
& $Python -m pytest -q

Write-Host "CPU development environment is ready."
Write-Host "Run .\run_gui_windows.bat to start HOMR."
