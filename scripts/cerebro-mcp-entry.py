#!/usr/bin/env python3
"""Cerebro MCP Server - Standalone entry point for PyInstaller bundle.

This entry point is used when building the MCP memory server as a
standalone binary for the Full Stack desktop edition.
"""
import os
import sys
import asyncio
import platform
from pathlib import Path


def get_data_dir():
    """Get platform-appropriate data directory."""
    # Allow override via environment
    if os.environ.get("CEREBRO_DATA_DIR"):
        return Path(os.environ["CEREBRO_DATA_DIR"])

    if platform.system() == "Windows":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return Path(base) / "Cerebro" / "data"
    else:
        return Path.home() / ".cerebro" / "data"


def setup_environment():
    """Set up embedded environment defaults."""
    data_dir = get_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    defaults = {
        "CEREBRO_DATA_DIR": str(data_dir),
        # Disable features that need large ML models (torch, transformers, faiss)
        # The bundled MCP server uses keyword search only to keep size manageable
        "CEREBRO_EMBEDDING_MODE": "keyword_only",
        "CEREBRO_LOG_LEVEL": "INFO",
    }

    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value


def main():
    setup_environment()

    # Add the bundled source to path
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)

    from mcp_ultimate_memory import main as mcp_main
    asyncio.run(mcp_main())


if __name__ == "__main__":
    main()
