#!/bin/bash
exec /home/maxime/Projects/mrmd-packages/mrmd-python/.venv/bin/python -c "
from mrmd_python.mcp_server import create_mcp_server
mcp = create_mcp_server()
mcp.run(transport='stdio')
"
