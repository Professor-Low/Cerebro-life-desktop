# Cerebro Desktop

A personal AI companion that Remembers and learns from every conversation you have avoiding the stagnation that normal AI agets have. Runs entirely on your machine containerized. Cerebro wraps a Docker-based backend (FastAPI + Redis) inside an Electron shell with system tray integration, delivering a Claude-code (Opas 4.6) powered orchestration layer with unified memory across all agent assistants. Cerebro will also not Rely on API keys as consuming expensive API credits is a problem that most agets are strugeling with today. It runs through your Claude CLI subscription and will expand to support additional local model options for users with more advanced hardware.

## Prerequisites

- **Docker Desktop** (Windows or Linux)
- **Claude Code subscription** (for MCP memory integration)

## Install

Download the latest installer from [GitHub Releases](https://github.com/Professor-Low/Cerebro-life-desktop/releases):

| Platform | File |
|----------|------|
| Windows  | `Cerebro-Installer.exe` |
| Linux    | `Cerebro-Installer.AppImage` or `.deb` |

Run the installer. On first launch Cerebro will pull the required Docker images automatically.

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

## Project Structure

```
electron/      Electron main process (main.js, managers, tray)
frontend/      Static frontend assets served inside the app
docker/        Docker Compose stack and container configs
backend-src/   Python backend source (FastAPI + cognitive loop)
memory-src/    MCP memory server source
build/         electron-builder configs
assets/        App icons and images
scripts/       Build and packaging scripts
```

## License

Cerebro Source Available License - See [LICENSE](LICENSE)
