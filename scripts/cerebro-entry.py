#!/usr/bin/env python3
"""Cerebro Desktop - Standalone entry point for PyInstaller bundle."""
import os
import sys
import secrets
import platform
from pathlib import Path


def get_data_dir():
    """Get platform-appropriate data directory."""
    if platform.system() == "Windows":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return Path(base) / "Cerebro"
    else:
        return Path.home() / ".config" / "cerebro"


def setup_environment():
    """Set up embedded environment defaults."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    # Set defaults only if not already set
    defaults = {
        "CEREBRO_STANDALONE": "1",
        "CEREBRO_HOST": "127.0.0.1",
        "CEREBRO_PORT": "59000",
        "REDIS_URL": "redis://localhost:16379/0",
        "AI_MEMORY_PATH": str(data_dir / "memory"),
        "CEREBRO_DATA_DIR": str(data_dir),
    }

    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value

    # Auto-generate JWT secret on first run
    secret_file = data_dir / ".jwt_secret"
    if secret_file.exists():
        os.environ.setdefault("CEREBRO_SECRET", secret_file.read_text().strip())
    else:
        secret = secrets.token_hex(32)
        secret_file.write_text(secret)
        os.environ.setdefault("CEREBRO_SECRET", secret)

    # Create memory directory
    (data_dir / "memory").mkdir(parents=True, exist_ok=True)


def main():
    setup_environment()

    # Add backend directory to path
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    import uvicorn
    from main import socket_app

    host = os.environ.get("CEREBRO_HOST", "127.0.0.1")
    port = int(os.environ.get("CEREBRO_PORT", "59000"))

    uvicorn.run(socket_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
