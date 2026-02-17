# Cerebro Memory Server

The core MCP (Model Context Protocol) memory server for Cerebro. Provides 49 tools for persistent memory, reasoning, and learning.

## Installation

```bash
pip install cerebro-ai
```

## Quick Start

```bash
# Initialize Cerebro
cerebro init

# Verify installation
cerebro doctor

# Start the MCP server (usually done automatically by Claude Code)
cerebro serve
```

## Features

- **3-Tier Memory**: Episodic (events), Semantic (facts), Working (active reasoning)
- **49 MCP Tools**: Comprehensive cognitive toolkit
- **Hybrid Search**: Semantic + keyword search with FAISS
- **Learning System**: Auto-detects and promotes patterns
- **Privacy-First**: Built-in secret detection and filtering
- **Cross-Platform**: Windows, Linux, macOS

## Configuration

See [.env.example](../.env.example) for all configuration options.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Documentation

- [Full Tool Reference](../docs/MCP_TOOLS.md)
- [Architecture](../docs/ARCHITECTURE.md)
