$ErrorActionPreference = "Stop"

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  ECGPredict — Starting Backend Server" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

$repoRoot = Split-Path -Parent $PSScriptRoot

$venvPy = Join-Path $repoRoot "ecg_env\Scripts\python.exe"
$sysPy = "python"

if (Test-Path $venvPy) {
    $pythonExe = $venvPy
} else {
    $pythonExe = $sysPy
}

Write-Host ("Python: " + $pythonExe) -ForegroundColor Green

$reqFile = Join-Path $PSScriptRoot "backend\requirements.txt"

Write-Host "Installing backend requirements..." -ForegroundColor Yellow
& $pythonExe -m pip install -r $reqFile -q

$modelPath = Join-Path $repoRoot "models\resnet1d.pth"

if (Test-Path $modelPath) {

    $sizeMB = [math]::Round((Get-Item $modelPath).Length / 1MB, 1)
    Write-Host ("Model found: resnet1d.pth (" + $sizeMB + " MB)") -ForegroundColor Green

} else {

    Write-Host ("[WARNING] Model file not found: " + $modelPath) -ForegroundColor Red
}

Write-Host ""
Write-Host "Starting FastAPI server..." -ForegroundColor Cyan
Write-Host "Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:8000/app" -ForegroundColor Cyan
Write-Host "Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop server" -ForegroundColor Gray
Write-Host "============================================`n" -ForegroundColor Cyan

Set-Location (Join-Path $PSScriptRoot "backend")

& $pythonExe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload