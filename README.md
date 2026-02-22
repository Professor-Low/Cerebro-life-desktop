# Cerebro Desktop

A personal AI companion that runs entirely on your machine. Cerebro wraps a Docker-based backend (FastAPI + Ollama + Redis) inside an Electron shell with system-tray integration, giving you a local AI assistant with no cloud dependency.

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
