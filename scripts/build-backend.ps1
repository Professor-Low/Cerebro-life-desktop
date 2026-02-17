# Build Cerebro backend as standalone executable (Windows)
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ProjectDir = Split-Path -Parent $ScriptDir
$BackendDir = Join-Path $ProjectDir "backend-src"

Write-Host "Building Cerebro backend..."
Write-Host "  ScriptDir:  $ScriptDir"
Write-Host "  ProjectDir: $ProjectDir"
Write-Host "  BackendDir: $BackendDir"

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

# PyInstaller with spec files may ignore --distpath and output to cwd\dist\ instead.
# Ensure the output ends up where we expect it.
$ExpectedOutput = Join-Path $ProjectDir "backend\cerebro-server"
if (-not (Test-Path $ExpectedOutput)) {
    $FallbackOutput = Join-Path $BackendDir "dist\cerebro-server"
    if (Test-Path $FallbackOutput) {
        Write-Host "Output not at expected --distpath location, moving from dist\..."
        New-Item -ItemType Directory -Path (Join-Path $ProjectDir "backend") -Force | Out-Null
        Move-Item $FallbackOutput (Join-Path $ProjectDir "backend\cerebro-server")
    } else {
        Write-Error "Could not find PyInstaller output"
        exit 1
    }
}

# Clean up
Remove-Item -Force "$BackendDir\cerebro-entry.py" -ErrorAction SilentlyContinue
deactivate

Write-Host "Backend build complete: $ProjectDir\backend\cerebro-server\"
