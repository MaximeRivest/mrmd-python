"""
mrmd-python: MRP runtime server for Python.

Primary interface is MCP, served at /mcp.
"""

from .service import RuntimeService
from .mcp_server import create_mcp_server
from .app import create_app
from .worker import IPythonWorker

__version__ = "0.4.0"
__all__ = ["RuntimeService", "create_mcp_server", "create_app", "IPythonWorker"]
