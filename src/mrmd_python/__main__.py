"""Run mrmd-python as an MCP server."""

import argparse
import os
import sys

import uvicorn


def main():
    parser = argparse.ArgumentParser(
        prog="mrmd-python",
        description="MRP runtime server for Python (MCP-native)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--cwd", default=None, help="Working directory")
    parser.add_argument("--venv", default=None, help="Virtual environment path")
    parser.add_argument("--assets-dir", default=None, help="Asset storage directory")

    args = parser.parse_args()

    os.environ.setdefault("MRMD_CWD", args.cwd or os.getcwd())
    if args.venv:
        os.environ["MRMD_VENV"] = args.venv
    if args.assets_dir:
        os.environ["MRMD_ASSETS_DIR"] = args.assets_dir

    print(f"mrmd-python 0.4.0")
    print(f"  MCP endpoint: http://{args.host}:{args.port}/mcp-server/mcp")
    print(f"  Health:       http://{args.host}:{args.port}/health")
    print(f"  Assets:       http://{args.host}:{args.port}/assets/{{id}}")
    print()

    uvicorn.run(
        "mrmd_python.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
