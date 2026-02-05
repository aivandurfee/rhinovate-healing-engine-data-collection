# Rhinovate Data Collection – Quick Start

## One-time setup

```powershell
.\setup.ps1
```

This creates a `.venv` with Python 3.9+ and installs all dependencies (selenium, mediapipe, opencv, etc.).

## Run commands

Use `run.ps1` or `run.bat` so scripts always use the project venv:

```powershell
# ASPS scraper (10 patients)
.\run.ps1 scripts/asps_download.py --max-cases 10

# Process and organize (side profiles only, landmark-based detection)
.\run.ps1 scripts/process_and_organize.py

# Full pipeline
.\run.ps1 scripts/asps_download.py --max-cases 10
.\run.ps1 scripts/process_and_organize.py
```

On Windows CMD:

```cmd
run.bat scripts\asps_download.py --max-cases 10
```

## Git Bash

```bash
./setup.sh
./run.sh scripts/asps_download.py --max-cases 10
./run.sh scripts/process_and_organize.py
```

## If PowerShell blocks scripts

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run `.\setup.ps1` and `.\run.ps1` as above.
