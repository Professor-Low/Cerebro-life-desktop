# Cerebro Backend

FastAPI backend with cognitive loop engine, real-time streaming, and agent orchestration.

## Features

- **OODA Cognitive Loop**: Observe-Orient-Decide-Act-Reflect-Learn cycle
- **Real-time Streaming**: Socket.IO for live thought narration
- **Agent Orchestration**: Multi-agent task decomposition and parallel execution
- **Browser Automation**: Self-healing locators with Playwright
- **Skill Generation**: Voyager-pattern skill learning and reuse

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8400 --reload
```

## Configuration

See [.env.example](../.env.example) for all configuration options.

## API Documentation

Once running, visit `http://localhost:8400/docs` for the interactive API documentation.

## Architecture

See [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) for details on the cognitive loop and OODA engine.
