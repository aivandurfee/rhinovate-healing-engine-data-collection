# Rhinovate – Run scripts with the project venv (creates venv if needed)
# Usage: .\run.ps1 scripts/asps_download.py --max-cases 10
#        .\run.ps1 scripts/process_and_organize.py --asps-mode

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Running setup first (creates venv, installs deps)..."
    & (Join-Path $ProjectRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

$script = $args[0]
if (-not $script) {
    Write-Host "Usage: .\run.ps1 <script> [args...]"
    Write-Host "Example: .\run.ps1 scripts/asps_download.py --max-cases 10"
    exit 1
}

$scriptPath = Join-Path $ProjectRoot $script
if (-not (Test-Path $scriptPath)) {
    Write-Host "Script not found: $scriptPath" -ForegroundColor Red
    exit 1
}

& $VenvPython $scriptPath @args[1..($args.Length-1)]
