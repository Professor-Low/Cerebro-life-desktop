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

# PyInstaller with spec files may ignore --distpath and output to cwd/dist/ instead.
# Ensure the output ends up where we expect it.
if [ ! -d "$PROJECT_DIR/backend/cerebro-server" ]; then
  echo "Output not at expected --distpath location, checking dist/..."
  if [ -d "$BACKEND_DIR/dist/cerebro-server" ]; then
    mkdir -p "$PROJECT_DIR/backend"
    mv "$BACKEND_DIR/dist/cerebro-server" "$PROJECT_DIR/backend/"
    echo "Moved output from dist/ to backend/"
  else
    echo "ERROR: Could not find PyInstaller output"
    find "$BACKEND_DIR" -name "cerebro-server" -type d 2>/dev/null || true
    exit 1
  fi
fi

# Clean up
rm -f "$BACKEND_DIR/cerebro-entry.py"
deactivate

echo "Backend build complete: $PROJECT_DIR/backend/cerebro-server/"
ls -la "$PROJECT_DIR/backend/cerebro-server/" | head -5
