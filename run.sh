#!/usr/bin/env bash
# Rhinovate – Run scripts with project venv (use with Git Bash)
# Usage: ./run.sh scripts/asps_download.py --max-cases 10

set -e
cd "$(dirname "$0")"
VENV=".venv"

# Setup if venv missing
if [ ! -f "$VENV/Scripts/python.exe" ] && [ ! -f "$VENV/bin/python" ]; then
  echo "Running setup first..."
  ./setup.sh
fi

if [ -f "$VENV/Scripts/python.exe" ]; then
  PY="$VENV/Scripts/python.exe"
else
  PY="$VENV/bin/python"
fi

if [ -z "$1" ]; then
  echo "Usage: ./run.sh <script> [args...]"
  echo "Example: ./run.sh scripts/asps_download.py --max-cases 10"
  exit 1
fi

SCRIPT="$1"
shift
"$PY" "$SCRIPT" "$@"
