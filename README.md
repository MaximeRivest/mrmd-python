# rat-py

**Run AnyThing: Python.** MCP server for live Python execution. 3 tools, persistent state, real-time streaming.

Part of the [rat](https://github.com/maximerivest) family: `rat-py`, `rat-r`, `rat-ju`, `rat-sh`, `rat-js` — the same 3 tools for every REPL language, all driven by [mcp2cli](https://github.com/maximerivest/mcp2cli).

## Install

```bash
pip install rat-py
rat-py install
```

That registers with [mcp2cli](https://github.com/maximerivest/mcp2cli) as `py`. Done.

## Use

```bash
mcp2cli py up                          # start persistent runtime

py run 'x = 42; print(x)'             # → 42  ✓ 3ms | 1 var
py run 'print(f"x is {x}")'           # → x is 42  ✓ 2ms | 1 var
py look                                # → python idle | 1 vars | exec #2
py look --at x                         # → x: int = 42
py ctl --op reset                      # → RESET | namespace cleared | 0 vars

mcp2cli py down                        # stop
```

Everything after `mcp2cli py up` goes through [mcp2cli](https://github.com/maximerivest/mcp2cli). The Python package is just the MCP server — mcp2cli handles the CLI, background process, shell completions, and output formatting.

## Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `run` | Run code or provide input | `py run 'df.head()'` |
| `look` | See variables, inspect, complete | `py look` or `py look --at df` |
| `ctl` | Reset, cancel, restart, status | `py ctl --op reset` |

### `run` — execute code

```
py run 'for i in range(3): print(i)'
→ 0
  1
  2
  ✓ 12ms | 1 var
```

Streams stdout/stderr in real-time via MCP notifications. Handles `input()` calls:

```
py run 'name = input("Name: ")'       → INPUT REQUESTED: "Name: "
py run --input Alice                    → ✓ 50ms | 1 var
```

### `look` — see state, inspect, drill down, complete

**Overview:**
```
py look
→ python idle | 3 vars | exec #5

  data   dict   {'name': 'Alice', 'scores': [90, 85, 92]}
  x      int    42
  model  Model  <Model object at 0x...>
```

**Inspect and drill down** — dot syntax navigates dicts, lists, and object attributes:
```
py look --at data
→ data: dict (3 items)
    = {'name': 'Alice', 'scores': [90, 85, 92], 'info': {'age': 30}}

      'name'    str   'Alice'
    ▸ 'scores'  list  [90, 85, 92]
    ▸ 'info'    dict  {'age': 30}

py look --at data.scores
→ data.scores: list (3 items)
    = [90, 85, 92]

      [0]  int  90
      [1]  int  85
      [2]  int  92

py look --at model
→ model: Model
    = <Model object at 0x...>

      lr      float  0.01
    ▸ layers  list   [{'type': 'dense', ...}, ...]
```

`▸` marks expandable children — keep drilling with `py look --at model.layers.0`.

**Functions and modules:**
```
py look --at math.sqrt
→ math.sqrt: builtin_function_or_method (function)
    Return the square root of x.
```

**Completions** — live from runtime state:
```
py look --code 'math.sq' --cursor 7
→ sqrt  function  sqrt(x, /)
```

### `ctl` — control

```
py ctl --op reset                      # clear namespace
py ctl --op reset --scope all          # clear everything
py ctl --op cancel                     # cancel running execution
py ctl --op restart                    # restart interpreter
py ctl --op status                     # python idle | exec #5 | rev 17 | up 3600s
```

## mcp2cli features you get for free

Because rat-py is just an MCP server, [mcp2cli](https://github.com/maximerivest/mcp2cli) gives you all of this automatically:

```bash
mcp2cli py up                          # background daemon, instant responses
mcp2cli py down                        # stop
mcp2cli py tools                       # list tools
mcp2cli py tools run                   # inspect a tool's schema
mcp2cli py shell                       # interactive mode with history + tab completion
mcp2cli py doctor                      # diagnose problems
py run 'x = 42' -o json               # JSON output for scripts
```

## Connect from other hosts

### Claude Desktop

```json
{
  "mcpServers": {
    "python": {
      "command": "rat-py",
      "args": ["--cwd", "/path/to/project"]
    }
  }
}
```

### Cursor

```json
{
  "mcpServers": {
    "python": {
      "command": "rat-py"
    }
  }
}
```

### MCP Inspector

```bash
rat-py --http --port 8000
# Inspector → Transport: Streamable HTTP → URL: http://127.0.0.1:8000/mcp
```

### Programmatic

```python
from rat_py import create_mcp_server

mcp = create_mcp_server()
mcp.run()  # STDIO
```

## For notebook GUIs

Same 3 tools serve notebook UIs through MCP:

- **Real-time output**: `run` streams stdout/stderr as MCP logging notifications
- **Completions**: `look(code=buffer, cursor=pos)` returns typed completion items
- **Hover/inspect**: `look(at=symbol)` returns type, value, docstring
- **Variable explorer**: `look()` returns variable table; `look(at="df")` shows `▸` expandable children
- **Drill-down**: `look(at="data.scores.0")` navigates dicts, lists, objects

## Features

- **Persistent state** — variables survive between executions
- **Real-time streaming** — stdout/stderr via MCP notifications (tqdm works)
- **Live completions** — from actual runtime state
- **Symbol inspection** — type, value, signature, docstring
- **Variable drill-down** — navigate dicts, lists, objects via dot syntax
- **Interactive input** — `input()` calls become `run(input="...")` responses
- **Asset generation** — matplotlib figures, HTML output served at `/assets/{id}`
- **Auto venv detection** — uses current venv or `VIRTUAL_ENV`

## Architecture

```
MCP Client (mcp2cli / Claude Desktop / Cursor / notebook GUI)
    │
    ├── run(code="...")          ← execute, stream output
    ├── look(at="df")           ← inspect, complete, overview
    └── ctl(op="reset")         ← control
    │
    ▼
FastMCP Server (mcp_server.py)
    │
    ▼
RuntimeService (service.py)     ← execution lifecycle, event log
    │
    ▼
IPythonWorker (worker.py)       ← Python execution engine (subprocess)
```

## The rat family

```
Any language runtime
    ↓ wrap in MCP (rat-py, rat-r, rat-ju, ...)
MCP Server
    ↓ automatic
├── mcp2cli  → CLI
├── mcp2py   → Python library
├── mcp2r    → R library
└── any MCP host → Claude Desktop, Cursor, notebooks
```

Same 3 tools — `run`, `look`, `ctl` — for every language. One interface to learn, works everywhere.
