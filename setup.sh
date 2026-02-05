#!/usr/bin/env bash
# Rhinovate – One-time setup (use with Git Bash)
# Run: ./setup.sh  or  bash setup.sh

set -e
cd "$(dirname "$0")"
VENV=".venv"
REQUIREMENTS="requirements.txt"

# Find Python 3.9+
for py in python3.14 python3.12 python3.11 python3.10 python3.9 python3 python; do
  if $py -c 'import sys; exit(0 if sys.version_info >= (3,9) else 1)' 2>/dev/null; then
    PYTHON=$py
    echo "Using: $($py --version)"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Python 3.9+ required. Install from https://www.python.org/downloads/"
  exit 1
fi

# Create venv if needed
if [ ! -f "$VENV/Scripts/python.exe" ] && [ ! -f "$VENV/bin/python" ]; then
  echo "Creating virtual environment..."
  $PYTHON -m venv "$VENV"
fi

# Use venv's python directly (avoids pip path issues on Windows)
if [ -f "$VENV/Scripts/python.exe" ]; then
  PIP="$VENV/Scripts/python.exe -m pip"
else
  PIP="$VENV/bin/python -m pip"
fi

$PIP install --upgrade pip -q
$PIP install -r "$REQUIREMENTS"

echo ""
echo "Setup complete. Use:"
echo "  ./run.sh scripts/asps_download.py --max-cases 10"
echo "  ./run.sh scripts/process_and_organize.py --asps-mode"
echo ""
