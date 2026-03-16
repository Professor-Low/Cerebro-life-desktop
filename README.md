# Cerebro Desktop

A personal AI companion that remembers and learns from every conversation, avoiding the stagnation that normal AI agents have. Runs entirely on your machine — no Docker, no containers, no API keys.

Cerebro wraps a native backend (FastAPI + Redis) inside an Electron shell with system tray integration, delivering a Claude Code (Opus 4.6) powered orchestration layer with unified memory across all agent assistants. Cerebro runs through your Claude CLI subscription and will expand to support additional local model options for users with more advanced hardware.

## Prerequisites

- **Claude Code subscription** (for MCP memory integration)
- **Python 3.10+** (bundled in release builds via PyInstaller)
- **Redis** (auto-installed on first launch)

## Install

Download the latest installer from [GitHub Releases](https://github.com/Professor-Low/Cerebro-life-desktop/releases):

| Platform | File |
|----------|------|
| Windows  | `Cerebro-Installer.exe` |
| Linux    | `Cerebro-Installer.AppImage` or `.deb` |

Run the installer. On first launch Cerebro will set up the native backend automatically — no Docker required.

## Build from Source

```bash
# Install dependencies
npm ci

# Build for Windows
npm run build:win

# Build for Linux
npm run build:linux
```

Installers are written to the `dist/` directory.

## Architecture (v4.0+)

Cerebro uses a fully native architecture — no Docker dependency:

- **NativeManager** (`electron/native-manager.js`) — manages Redis + FastAPI backend as native processes
- **PyInstaller backend** — Python backend bundled as a standalone executable in release builds
- **Redis** — auto-downloaded and managed per-platform
- **Electron** — desktop shell with system tray, auto-updates, and MCP integration

## Project Structure

```
electron/      Electron main process (main.js, managers, tray)
frontend/      Static frontend assets served inside the app
backend-src/   Python backend source (FastAPI + cognitive loop)
memory-src/    MCP memory server source
config/        MCP and runtime configuration
build/         electron-builder configs
assets/        App icons and images
scripts/       Build and packaging scripts
```

## License

Cerebro Source Available License — See [LICENSE](LICENSE)
