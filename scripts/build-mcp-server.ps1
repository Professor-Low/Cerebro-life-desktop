# Build Cerebro MCP memory server as standalone executable (Windows)
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$MemoryDir = Join-Path $ProjectDir "memory-src"

Write-Host "Building Cerebro MCP server..."

# Create/activate venv
python -m venv "$ScriptDir\.build-venv"
& "$ScriptDir\.build-venv\Scripts\Activate.ps1"

# Install MCP server dependencies (core only, no embeddings)
pip install mcp anyio numpy pydantic python-dateutil httpx
pip install pyinstaller

# Run PyInstaller
Push-Location "$MemoryDir\src"
pyinstaller "$ScriptDir\cerebro-mcp.spec" --distpath "$ProjectDir\mcp-server" --workpath "$ScriptDir\.build-work" --clean
Pop-Location

deactivate

Write-Host "MCP server build complete: $ProjectDir\mcp-server\cerebro-mcp-server\"
