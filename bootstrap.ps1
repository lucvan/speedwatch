# bootstrap.ps1 — fetch the gitignored bits needed to run speedwatch from a clean clone.
# Creates the venv, installs deps, downloads the YOLO11 model, and fetches ffmpeg/ffprobe.
# Re-runnable: skips anything already present. Run from the repo root.

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

# 1) Python venv + deps -------------------------------------------------------
$Venv = Join-Path $Root ".venv"
if (-not (Test-Path $Venv)) {
    Write-Host "Creating virtualenv..."
    python -m venv $Venv
}
$Py = Join-Path $Venv "Scripts\python.exe"
Write-Host "Installing requirements (onnxruntime-directml is reinstalled last)..."
& $Py -m pip install --upgrade pip
& $Py -m pip install -r (Join-Path $Root "requirements.txt")
# ultralytics may pull plain onnxruntime during export; force directml back so
# DmlExecutionProvider is available (see requirements.txt note).
& $Py -m pip install --force-reinstall --no-deps onnxruntime-directml==1.24.4

# 2) YOLO11 model -------------------------------------------------------------
$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force $Models | Out-Null
$Onnx = Join-Path $Models "yolo11m.onnx"
if (-not (Test-Path $Onnx)) {
    Write-Host "Exporting yolo11m.onnx via ultralytics..."
    # Downloads yolo11m.pt then exports to ONNX in the models/ dir.
    & $Py -c "from ultralytics import YOLO; YOLO('yolo11m.pt').export(format='onnx')"
    if (Test-Path (Join-Path $Root "yolo11m.onnx")) {
        Move-Item -Force (Join-Path $Root "yolo11m.onnx") $Onnx
    }
} else {
    Write-Host "yolo11m.onnx already present, skipping."
}

# 3) ffmpeg / ffprobe ---------------------------------------------------------
$Ffmpeg = Join-Path $Root "ffmpeg.exe"
if (-not (Test-Path $Ffmpeg)) {
    Write-Host "Downloading ffmpeg (gyan.dev essentials build)..."
    $Zip = Join-Path $env:TEMP "ffmpeg-speedwatch.zip"
    $Dir = Join-Path $env:TEMP "ffmpeg-speedwatch"
    Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $Zip
    if (Test-Path $Dir) { Remove-Item -Recurse -Force $Dir }
    Expand-Archive -Path $Zip -DestinationPath $Dir
    $bin = Get-ChildItem -Path $Dir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
    Copy-Item $bin.FullName $Ffmpeg
    Copy-Item (Join-Path $bin.DirectoryName "ffprobe.exe") (Join-Path $Root "ffprobe.exe")
    Remove-Item -Force $Zip; Remove-Item -Recurse -Force $Dir
} else {
    Write-Host "ffmpeg.exe already present, skipping."
}

Write-Host ""
Write-Host "Bootstrap complete. Next: copy .env.example to .env, fill it in, then run .\install-service.ps1"
