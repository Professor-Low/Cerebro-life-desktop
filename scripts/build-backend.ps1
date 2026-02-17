# Build Cerebro backend as standalone executable (Windows)
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $ProjectDir "backend-src"

Write-Host "Building Cerebro backend..."

# Create/activate venv
python -m venv "$ScriptDir\.build-venv"
& "$ScriptDir\.build-venv\Scripts\Activate.ps1"

# Install dependencies
pip install -r "$BackendDir\requirements.txt"
pip install pyinstaller

# Copy entry point to backend dir (PyInstaller needs it there)
Copy-Item "$ScriptDir\cerebro-entry.py" "$BackendDir\cerebro-entry.py"

# Run PyInstaller
Push-Location $BackendDir
pyinstaller "$ScriptDir\cerebro-server.spec" --distpath "$ProjectDir\backend" --workpath "$ScriptDir\.build-work" --clean
Pop-Location

# Clean up
Remove-Item -Force "$BackendDir\cerebro-entry.py" -ErrorAction SilentlyContinue
deactivate

Write-Host "Backend build complete: $ProjectDir\backend\cerebro-server\"
