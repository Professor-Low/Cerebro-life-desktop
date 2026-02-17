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
pip install mcp anyio numpy pydantic python-dateutil httpx
pip install pyinstaller

# Run PyInstaller
cd "$MEMORY_DIR/src"
pyinstaller "$SCRIPT_DIR/cerebro-mcp.spec" --distpath "$PROJECT_DIR/mcp-server" --workpath "$SCRIPT_DIR/.build-work" --clean

deactivate

echo "MCP server build complete: $PROJECT_DIR/mcp-server/cerebro-mcp-server/"
