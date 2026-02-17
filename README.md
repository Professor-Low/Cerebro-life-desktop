# Cerebro Desktop

Desktop application for [Cerebro](https://cerebro.life) - Your AI, Everywhere.

Built with Electron 34, featuring local AI backend management, system tray integration, and MCP memory server bundling.

## Editions

### Full Stack (~350-400 MB)
Everything you need in one installer:
- Electron desktop shell
- Python backend server (FastAPI + Socket.IO)
- Bundled Redis
- MCP memory server (49 tools for Claude Code integration)

Best for new users who want a complete self-contained setup.

### Client (~80 MB)
Lightweight Electron shell that connects to a remote Cerebro server.

Best for users who already have Cerebro/MCP running on a server or another machine.

## Development

```bash
# Install dependencies
npm install

# Run in dev mode (local backend from source)
npm run start:dev

# Run in remote mode
npm run start:remote
```

## Building

```bash
# Build the backend binary (required for Full Stack)
npm run build:backend

# Build Full Stack edition
npm run build:fullstack:linux
npm run build:fullstack:win

# Build Client edition
npm run build:client:linux
npm run build:client:win
```

## Project Structure

```
electron/          Electron main process (main.js, managers, tray)
frontend/          Static frontend assets served by backend
assets/            App icons and images
build/             electron-builder configs per edition
scripts/           Build scripts (PyInstaller specs, shell scripts)
backend-src/       Python backend source (vendored for builds)
memory-src/        MCP memory server source (vendored for builds)
```

## License

AGPL-3.0 - Copyright 2024-2026 Professor (Michael Anthony Lopez)
