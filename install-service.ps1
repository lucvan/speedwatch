# Run as Administrator to install/update the speedwatch NSSM service.
# Usage: .\install-service.ps1   (re-run to update parameters after changes)

$ServiceName = "speedwatch"
$InstallDir  = "C:\speedwatch"
$Python      = "$InstallDir\.venv\Scripts\python.exe"
$LogDir      = "$InstallDir\data\logs"

New-Item -ItemType Directory -Force $LogDir | Out-Null

$existing = nssm status $ServiceName 2>&1
if ($existing -notmatch "can't open service") {
    Write-Host "Stopping existing service..."
    nssm stop $ServiceName confirm
    nssm remove $ServiceName confirm
}

Write-Host "Installing $ServiceName service..."
nssm install $ServiceName $Python "-m" "app.main"
nssm set $ServiceName AppDirectory $InstallDir
nssm set $ServiceName AppStdout    "$LogDir\stdout.log"
nssm set $ServiceName AppStderr    "$LogDir\stderr.log"
nssm set $ServiceName AppRotateFiles 1
nssm set $ServiceName AppRotateSeconds 86400
nssm set $ServiceName AppRotateBytes 10485760
nssm set $ServiceName Start SERVICE_AUTO_START
nssm set $ServiceName Description "Speedwatch vehicle speed detection (standalone, YOLO11/DirectML)"

# Run as desktop user so DirectML can access GPU (session-0 / SYSTEM may be denied DML).
# Replace DOMAIN\username with your actual Windows user if not running as current user.
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
nssm set $ServiceName ObjectName $currentUser

Write-Host ""
Write-Host "Service installed. To start: nssm start $ServiceName"
Write-Host "UI: http://localhost:8765"
Write-Host ""
Write-Host "NOTE: You will be prompted for your Windows password to set the service account."
