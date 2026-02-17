#!/bin/bash
# Build Cerebro backend as standalone executable
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend-src"

echo "Building Cerebro backend..."

# Create/activate venv
python3 -m venv "$SCRIPT_DIR/.build-venv"
source "$SCRIPT_DIR/.build-venv/bin/activate"

# Install dependencies
pip install -r "$BACKEND_DIR/requirements.txt"
pip install pyinstaller

# Copy entry point to backend dir (PyInstaller needs it there)
cp "$SCRIPT_DIR/cerebro-entry.py" "$BACKEND_DIR/cerebro-entry.py"

# Run PyInstaller
cd "$BACKEND_DIR"
pyinstaller "$SCRIPT_DIR/cerebro-server.spec" --distpath "$PROJECT_DIR/backend" --workpath "$SCRIPT_DIR/.build-work" --clean

# Clean up
rm -f "$BACKEND_DIR/cerebro-entry.py"
deactivate

echo "Backend build complete: $PROJECT_DIR/backend/cerebro-server/"
