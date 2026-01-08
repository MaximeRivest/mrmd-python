# mrmd-python

Python runtime server implementing the MRMD Runtime Protocol (MRP).

## Installation

```bash
uv pip install mrmd-python
```

## Usage

### Command Line

```bash
mrmd-python --port 8000
```

### Programmatic

```python
from mrmd_python import create_app
import uvicorn

app = create_app(cwd="/path/to/project")
uvicorn.run(app, host="localhost", port=8000)
```

## API Endpoints

All endpoints are prefixed with `/mrp/v1/`.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/capabilities` | GET | Runtime capabilities |
| `/sessions` | GET/POST | List/create sessions |
| `/execute` | POST | Run code |
| `/execute/stream` | POST | Run code with SSE streaming |
| `/complete` | POST | Get completions |
| `/inspect` | POST | Get symbol info |
| `/hover` | POST | Get hover tooltip |
| `/variables` | POST | List variables |
| `/interrupt` | POST | Cancel execution |

## Protocol

See [PROTOCOL.md](../mrmd-editor/PROTOCOL.md) for the full MRP specification.
