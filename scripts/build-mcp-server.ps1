# Build Cerebro MCP memory server as standalone executable (Windows)
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectDir = Split-Path -Parent $ScriptDir
$MemoryDir = Join-Path $ProjectDir "memory-src"

Write-Host "Building Cerebro MCP server..."
Write-Host "  ScriptDir:  $ScriptDir"
Write-Host "  ProjectDir: $ProjectDir"
Write-Host "  MemoryDir:  $MemoryDir"

# Create/activate venv
python -m venv "$ScriptDir\.build-venv"
& "$ScriptDir\.build-venv\Scripts\Activate.ps1"

# Install MCP server dependencies (core only, no embeddings)
pip install "mcp[cli]" anyio numpy pydantic python-dateutil httpx
pip install pyinstaller

# Run PyInstaller
Push-Location "$MemoryDir\src"
pyinstaller "$ScriptDir\cerebro-mcp.spec" --distpath "$ProjectDir\mcp-server" --workpath "$ScriptDir\.build-work" --clean
Pop-Location

# PyInstaller with spec files may ignore --distpath and output to cwd\dist\ instead.
$ExpectedOutput = Join-Path $ProjectDir "mcp-server\cerebro-mcp-server"
if (-not (Test-Path $ExpectedOutput)) {
    $FallbackOutput = Join-Path $MemoryDir "src\dist\cerebro-mcp-server"
    if (Test-Path $FallbackOutput) {
        Write-Host "Output not at expected --distpath location, moving from dist\..."
        New-Item -ItemType Directory -Path (Join-Path $ProjectDir "mcp-server") -Force | Out-Null
        Move-Item $FallbackOutput (Join-Path $ProjectDir "mcp-server\cerebro-mcp-server")
    } else {
        Write-Error "Could not find PyInstaller output"
        exit 1
    }
}

deactivate

Write-Host "MCP server build complete: $ProjectDir\mcp-server\cerebro-mcp-server\"
