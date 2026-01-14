# mrmd-python

Independent Python runtime for MRMD notebooks. Runs as a daemon process with full GPU memory isolation.

## Features

- **Independent daemon process** - survives if parent dies, variables persist
- **GPU memory isolation** - kill daemon to release VRAM (critical for vLLM)
- **Auto venv detection** - uses current venv or `VIRTUAL_ENV`
- **Registry-based discovery** - find runtimes via `~/.mrmd/runtimes/`
- **Full MRP protocol** - execute, completions, inspect, variables, streaming

## Installation

```bash
# With uv
uv pip install mrmd-python

# Or run directly without installing
uvx mrmd-python
```

## Quick Start

```bash
# Start a daemon runtime (auto-detects venv)
mrmd-python

# The daemon runs in background. Use the API:
curl http://localhost:PORT/mrp/v1/capabilities

# List running runtimes
mrmd-python --list

# Kill when done (releases GPU memory)
mrmd-python --kill default
```

## CLI Reference

```bash
# Start daemon
mrmd-python                     # Start with ID "default"
mrmd-python --id vllm           # Start with custom ID
mrmd-python --venv /path/venv   # Use specific venv
mrmd-python --port 8000         # Use specific port

# Management
mrmd-python --list              # List all running runtimes
mrmd-python --info ID           # Get runtime details
mrmd-python --kill ID           # Kill a runtime
mrmd-python --kill-all          # Kill all runtimes

# Development
mrmd-python --foreground        # Run in foreground (no daemon)
```

## Virtual Environment Detection

When `--venv` is not specified, mrmd-python auto-detects:

1. Current venv (if running inside one via `sys.prefix`)
2. `VIRTUAL_ENV` environment variable
3. Falls back to system Python

## API Endpoints

All endpoints at `/mrp/v1/`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/capabilities` | GET | Runtime info and features |
| `/sessions` | GET/POST | List/create sessions |
| `/sessions/{id}` | GET/DELETE | Get/destroy session |
| `/execute` | POST | Run code |
| `/execute/stream` | POST | Run code with SSE streaming |
| `/complete` | POST | Get completions |
| `/inspect` | POST | Get symbol documentation |
| `/hover` | POST | Get hover tooltip |
| `/variables` | POST | List user variables |
| `/variables/{name}` | POST | Get variable details |
| `/is_complete` | POST | Check if code is complete |
| `/format` | POST | Format code with black |

## Architecture

```
~/.mrmd/
├── runtimes/
│   └── {id}.json    # Registry: pid, port, url, venv, cwd
└── logs/
    └── {id}.log     # Daemon logs
```

Each runtime is a fully independent process:
- Double-forked daemon (survives parent death)
- Own IPython shell with persistent variables
- HTTP server on auto-assigned port
- Registered in `~/.mrmd/runtimes/` for discovery

## GPU Memory Management

For vLLM and other GPU workloads, memory is only released when the process dies:

```bash
# Load model in runtime
mrmd-python --id vllm
# ... use the model ...

# Release GPU memory
mrmd-python --kill vllm
```

## Programmatic Usage

```python
from mrmd_python import create_app
import uvicorn

# Create app (daemon_mode=True for use inside daemon)
app = create_app(
    cwd="/path/to/project",
    venv="/path/to/venv",
    daemon_mode=True,
)
uvicorn.run(app, host="localhost", port=8000)
```

## Protocol

See [PROTOCOL.md](../mrmd-editor/PROTOCOL.md) for the full MRP specification.
