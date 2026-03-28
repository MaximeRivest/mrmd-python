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
py_ctl(op="restart")              # restart interpreter (fixes corrupted state)
py_ctl(op="status")               # python idle | exec #5 | rev 17 | up 3600s
```

## Install

```bash
uv pip install mrmd-python
```

## Start

```bash
mrmd-python                  # MCP over stdio (for Claude Desktop, Cursor)
mrmd-python start            # shared runtime (background, multi-client)
mrmd-python stop             # stop the shared runtime
mrmd-python status           # show all running runtimes
mrmd-python install          # show config for Claude Desktop / Cursor / mcp2cli
```

### Shared mode (recommended for collaboration)

One runtime, many clients — terminal, LLM, and notebook all share the same namespace:

```bash
cd ~/my-project
mrmd-python start
# → Runtime 'my-project' started (pid 12345)
# → MCP: http://127.0.0.1:8717/mcp
```

Now from the terminal, Claude Desktop, Cursor, a notebook GUI — all connect to the same URL, all see the same variables.

### Multiple runtimes

Each runtime gets a name (defaults to the directory name) and its own port:

```bash
cd ~/analysis && mrmd-python start            # name: 'analysis', port: 8717
cd ~/ml-work && mrmd-python start             # name: 'ml-work', port: 8718
mrmd-python start --name scratch              # explicit name, auto port

mrmd-python status
# NAME       PID    PORT   STATE  EXECS  CWD
# analysis   12345  8717   idle   5      ~/analysis
# ml-work    12346  8718   idle   3      ~/ml-work
# scratch    12347  8719   idle   0      .

mrmd-python stop --name analysis
mrmd-python stop --name ml-work
mrmd-python stop --name scratch
```

Each runtime is fully isolated — different namespace, can use different venvs. Starting from the same directory with the same name is blocked (one runtime per name).

### Stdio mode (for single-client MCP hosts)

Default when no subcommand — exactly what MCP hosts expect:

```bash
mrmd-python                  # stdio
mrmd-python --cwd ~/project  # specific directory
mrmd-python --venv ~/p/.venv # specific venv
```

## Connect

### Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "python": {
      "command": "uvx",
      "args": ["mrmd-python", "--cwd", "/path/to/your/project"]
    }
  }
}
```

Or run `mrmd-python install` to generate the config for your current directory.

### Cursor

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "python": {
      "command": "uvx",
      "args": ["mrmd-python"]
    }
  }
}
```

### mcp2cli

```bash
mcp2cli add python --command 'mrmd-python --cwd .'
mcp2cli tool python py --code 'print("hello")'
mcp2cli tool python py_look
```

### MCP Inspector

```bash
mrmd-python --http --port 8000
# Inspector → Transport: Streamable HTTP → URL: http://127.0.0.1:8000/mcp
```

### Programmatic

```python
from mrmd_python import create_mcp_server

# STDIO (for MCP hosts)
mcp = create_mcp_server()
mcp.run()

# HTTP
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
- **Named runtimes** — run multiple isolated runtimes simultaneously
- **Shared state** — multiple MCP clients connect to the same runtime
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
