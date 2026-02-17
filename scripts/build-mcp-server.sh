#!/bin/bash
# Build Cerebro MCP memory server as standalone executable
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MEMORY_DIR="$PROJECT_DIR/memory-src"

echo "Building Cerebro MCP server..."

# Create/activate venv
python3 -m venv "$SCRIPT_DIR/.build-venv"
source "$SCRIPT_DIR/.build-venv/bin/activate"

# Install MCP server dependencies (core only, no embeddings)
pip install "mcp[cli]" anyio numpy pydantic python-dateutil httpx
pip install pyinstaller

# Run PyInstaller
cd "$MEMORY_DIR/src"
pyinstaller "$SCRIPT_DIR/cerebro-mcp.spec" --distpath "$PROJECT_DIR/mcp-server" --workpath "$SCRIPT_DIR/.build-work" --clean

# PyInstaller with spec files may ignore --distpath and output to cwd/dist/ instead.
if [ ! -d "$PROJECT_DIR/mcp-server/cerebro-mcp-server" ]; then
  echo "Output not at expected --distpath location, checking dist/..."
  if [ -d "$MEMORY_DIR/src/dist/cerebro-mcp-server" ]; then
    mkdir -p "$PROJECT_DIR/mcp-server"
    mv "$MEMORY_DIR/src/dist/cerebro-mcp-server" "$PROJECT_DIR/mcp-server/"
    echo "Moved output from dist/ to mcp-server/"
  else
    echo "ERROR: Could not find PyInstaller output"
    find "$MEMORY_DIR" -name "cerebro-mcp-server" -type d 2>/dev/null || true
    exit 1
  fi
fi

deactivate

echo "MCP server build complete: $PROJECT_DIR/mcp-server/cerebro-mcp-server/"
ls -la "$PROJECT_DIR/mcp-server/cerebro-mcp-server/" | head -5
