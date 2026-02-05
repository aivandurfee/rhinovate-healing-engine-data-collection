# Rhinovate – One-time setup: create venv, install dependencies
# Run: .\setup.ps1
# Then use: .\run.ps1 scripts/asps_download.py --max-cases 10

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"

# Find Python 3.9+ (try py launcher first, then python)
$PythonExe = $null
foreach ($cmd in @("py -3", "py -3.14", "py -3.12", "py -3.11", "py -3.10", "py -3.9", "python")) {
    try {
        $null = Invoke-Expression "$cmd --version" 2>$null
        $ver = Invoke-Expression "$cmd -c 'import sys; v=sys.version_info; print(f\"{v.major}.{v.minor}\")'" 2>$null
        $major = [int]($ver -split '\.')[0]
        $minor = [int]($ver -split '\.')[1]
        if ($major -ge 3 -and $minor -ge 9) {
            $PythonExe = ($cmd -split ' ')[0]
            Write-Host "Using: $cmd -> Python $ver"
            break
        }
    } catch {}
}

if (-not $PythonExe) {
    Write-Host "Python 3.9+ required. Install from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

# Create venv if needed
if (-not (Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    Write-Host "Creating virtual environment..."
    if ($PythonExe -eq "py") {
        & py -3 -m venv $VenvPath
    } else {
        & $PythonExe -m venv $VenvPath
    }
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "Venv creation failed." -ForegroundColor Red
    exit 1
}

# Upgrade pip and install requirements
Write-Host "Installing dependencies..."
& $VenvPython -m pip install --upgrade pip -q
& $VenvPip install -r $RequirementsPath
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Setup complete. Use:" -ForegroundColor Green
Write-Host "  .\run.ps1 scripts/asps_download.py --max-cases 10"
Write-Host "  .\run.ps1 scripts/process_and_organize.py --asps-mode"
Write-Host ""
