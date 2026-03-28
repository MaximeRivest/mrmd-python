# mrmd-python

MCP server for live Python execution. 3 tools, persistent state, real-time streaming.

## What it does

`mrmd-python` is an MCP server that gives you a live Python runtime. Run code, inspect variables, get completions — all through the Model Context Protocol.

```
py(code="x = 42; print(x)")    →  42  ✓ 3ms | 1 var
py_look(at="x")                 →  x: int = 42
py_look(code="math.sq", cursor=7) →  sqrt  function  sqrt(x, /)
py_ctl(op="reset")              →  RESET | namespace cleared | 0 vars
```

## Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `py` | Run code or provide input | `py(code="df.head()")` |
| `py_look` | See variables, inspect symbols, get completions | `py_look()` or `py_look(at="df")` |
| `py_ctl` | Reset or cancel | `py_ctl(op="reset")` |

### `py` — run code

```
py(code="for i in range(3): print(i)")
→ 0
  1
  2
  ✓ 12ms | 1 var
```

Streams stdout/stderr in real-time via MCP notifications. Handles `input()` calls:

```
py(code="name = input('Name: ')")  →  INPUT REQUESTED: "Name: "
py(input="Alice")                   →  ✓ 50ms | 1 var
```

### `py_look` — see state, inspect, drill down, complete

**Overview:**
```
py_look()
→ python idle | 3 vars | exec #5

  data   dict        {'name': 'Alice', 'scores': [90, 85, 92]}
  x      int         42
  model  Model       <Model object at 0x...>
```

**Inspect and drill down** — dot syntax navigates dicts, lists, and object attributes:
```
py_look(at="data")
→ data: dict (3 items)
    = {'name': 'Alice', 'scores': [90, 85, 92], 'info': {'age': 30}}

      'name'    str   'Alice'
    ▸ 'scores'  list  [90, 85, 92]
    ▸ 'info'    dict  {'age': 30}

py_look(at="data.scores")
→ data.scores: list (3 items)
    = [90, 85, 92]

      [0]  int  90
      [1]  int  85
      [2]  int  92

py_look(at="model")
→ model: Model
    = <Model object at 0x...>

      lr      float  0.01
    ▸ layers  list   [{'type': 'dense', ...}, ...]
```

`▸` marks expandable children — keep drilling with `py_look(at="model.layers.0")`.

**Functions and modules:**
```
py_look(at="math.sqrt")
→ math.sqrt: builtin_function_or_method (function)
    Return the square root of x.
```

**Completions** — live from runtime state:
```
py_look(code="math.sq", cursor=7)
→ sqrt  function  sqrt(x, /)
```

### `py_ctl` — control

```
py_ctl(op="reset")                 # clear namespace
py_ctl(op="reset", scope="all")   # clear everything
py_ctl(op="cancel")               # cancel running execution
```

## Installation

```bash
uv pip install mrmd-python
```

## Usage

### As an MCP server (STDIO)

```bash
# With FastMCP CLI
fastmcp run mrmd_python.mcp_server:create_mcp_server

# Or directly
python -c "from mrmd_python.mcp_server import create_mcp_server; create_mcp_server().run()"
```

### As an HTTP server

```bash
python -m mrmd_python --port 8000
```

Serves:
- `/mcp-server/mcp` — MCP Streamable HTTP endpoint
- `/health` — plain HTTP health check
- `/assets/{id}` — runtime-generated assets (plots, HTML)

### With mcp2cli

```bash
mcp2cli add mrmd-python --command 'fastmcp run mrmd_python.mcp_server:create_mcp_server'
mcp2cli tool mrmd-python py --code 'print("hello")'
mcp2cli tool mrmd-python py_look
```

### With MCP Inspector

1. Start the HTTP server: `python -m mrmd_python --port 8000`
2. In Inspector, set Transport Type to **Streamable HTTP**
3. URL: `http://127.0.0.1:8000/mcp-server/mcp`
4. Connect

### Programmatic

```python
from mrmd_python import create_mcp_server, RuntimeService

# Create and run
mcp = create_mcp_server()
mcp.run()  # STDIO

# Or with a shared service
service = RuntimeService(cwd="/path/to/project")
mcp = create_mcp_server(service=service)

# Or as a FastAPI app
from mrmd_python import create_app
app = create_app(cwd="/path/to/project")
# uvicorn.run(app, host="127.0.0.1", port=8000)
```

## For notebook GUIs

The same 3 tools serve notebook UIs through MCP:

- **Real-time output**: `py` streams stdout/stderr as MCP logging notifications during execution
- **Completions**: `py_look(code=buffer, cursor=pos)` returns typed completion items on every keystroke
- **Hover/inspect**: `py_look(at=symbol)` returns type, value, signature, docstring
- **Variable explorer**: `py_look()` returns the full variable table; `py_look(at="df")` drills into children with `▸` expandable markers
- **Drill-down**: `py_look(at="data.scores.0")` navigates dicts, lists, and object attributes via dot syntax

A notebook GUI is just an MCP client that listens to notifications for streaming and calls `py_look` for IDE features.

## Features

- **Persistent state** — variables survive between executions
- **Real-time streaming** — stdout/stderr via MCP notifications
- **Live completions** — from actual runtime state, not just static analysis
- **Symbol inspection** — type, value, signature, docstring, source location
- **Variable drill-down** — navigate dicts, lists, objects via `py_look(at="data.scores.0")`
- **Interactive input** — `input()` calls become `py(input="...")` responses
- **Asset generation** — matplotlib figures, HTML output served at `/assets/{id}`
- **Auto venv detection** — uses current venv or `VIRTUAL_ENV`

## Architecture

```
MCP Client (agent / notebook GUI / CLI)
    │
    ├── py(code="...")           ← run code, stream output
    ├── py_look(at="df")        ← inspect, complete, overview
    └── py_ctl(op="reset")      ← control
    │
    ▼
FastMCP Server (mcp_server.py)
    │
    ▼
RuntimeService (service.py)     ← execution lifecycle, event log, state
    │
    ▼
IPythonWorker (worker.py)       ← actual Python execution engine
```

## Protocol

See the [MRP specification](../spec/mrp-protocol.md) for the full protocol design.
