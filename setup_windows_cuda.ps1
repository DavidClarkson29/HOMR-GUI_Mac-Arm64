$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.12 -m venv .venv
}

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install -e .
& $Python -m pip uninstall -y onnxruntime
& $Python -m pip install "onnxruntime-gpu==1.24.1"
& $Python -m pip install pytest pyinstaller
& $Python -m pytest -q

Write-Host "Available ONNX Runtime providers:"
& $Python -c "import onnxruntime as ort; print(ort.get_available_providers())"
Write-Host "CUDA must appear in the provider list before GPU packaging."
