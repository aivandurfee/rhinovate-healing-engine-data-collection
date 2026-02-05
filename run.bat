@echo off
setlocal
REM Rhinovate - Run scripts with project venv
REM Usage: run.bat scripts\asps_download.py --max-cases 10

set "ROOT=%~dp0"
set "VENV=%ROOT%.venv"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo Running setup first...
    powershell -ExecutionPolicy Bypass -File "%ROOT%setup.ps1"
    if errorlevel 1 exit /b 1
)

if "%~1"=="" (
    echo Usage: run.bat script [args...]
    echo Example: run.bat scripts\asps_download.py --max-cases 10
    exit /b 1
)

set "SCRIPT=%~1"
shift
"%PY%" "%ROOT%%SCRIPT%" %*
exit /b %ERRORLEVEL%
