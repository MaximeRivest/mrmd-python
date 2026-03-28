"""
mrmd-python: MCP server for live Python execution.

    mrmd-python              # start MCP server (stdio)
    mrmd-python --http       # start MCP server (HTTP)
    mrmd-python install      # show config for Claude Desktop / Cursor / mcp2cli
"""

import argparse
import json
import os
import shutil
import sys


def _find_venv() -> str | None:
    """Auto-detect virtual environment."""
    # Already in a venv
    if sys.prefix != sys.base_prefix:
        return sys.prefix
    # VIRTUAL_ENV env var
    if os.environ.get("VIRTUAL_ENV"):
        return os.environ["VIRTUAL_ENV"]
    # .venv in cwd or parents
    cwd = os.getcwd()
    for d in [cwd, *[os.path.dirname(cwd)] * 5]:
        candidate = os.path.join(d, ".venv")
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "bin", "python")):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _find_command() -> str:
    """Find the best command to use in configs."""
    # If installed as a script, use that
    if shutil.which("mrmd-python"):
        return "mrmd-python"
    # Fallback: uvx
    return "uvx mrmd-python"


def _install(target: str | None):
    """Print config snippets for MCP hosts."""
    cmd = _find_command()
    use_uvx = "uvx" in cmd
    cwd = os.getcwd()
    venv = _find_venv()

    print("# mrmd-python MCP configuration\n")

    if target in (None, "claude", "claude-desktop"):
        print("## Claude Desktop")
        print("# Add to ~/Library/Application Support/Claude/claude_desktop_config.json")
        print("# or ~/.config/Claude/claude_desktop_config.json\n")
        entry = {
            "command": "uvx" if use_uvx else "mrmd-python",
        }
        if use_uvx:
            entry["args"] = ["mrmd-python", "--cwd", cwd]
        else:
            entry["args"] = ["--cwd", cwd]
        if venv:
            entry["args"].extend(["--venv", venv])
        config = {"mcpServers": {"python": entry}}
        print(json.dumps(config, indent=2))
        print()

    if target in (None, "cursor"):
        print("## Cursor")
        print("# Add to .cursor/mcp.json in your project root\n")
        entry = {
            "command": "uvx" if use_uvx else "mrmd-python",
        }
        if use_uvx:
            entry["args"] = ["mrmd-python"]
        else:
            entry["args"] = []
        config = {"mcpServers": {"python": entry}}
        print(json.dumps(config, indent=2))
        print()

    if target in (None, "mcp2cli"):
        print("## mcp2cli")
        if use_uvx:
            print(f"mcp2cli add python --command 'uvx mrmd-python --cwd {cwd}'")
        else:
            print(f"mcp2cli add python --command 'mrmd-python --cwd {cwd}'")
        print()

    if target in (None, "inspector"):
        print("## MCP Inspector")
        print("# 1. Start HTTP server:")
        print(f"#    mrmd-python --http --port 8000 --cwd {cwd}")
        print("# 2. In Inspector: Transport = Streamable HTTP")
        print("#    URL = http://127.0.0.1:8000/mcp-server/mcp")
        print()

    if target in (None, "json"):
        print("## Generic MCP JSON config")
        entry = {"command": "uvx" if use_uvx else "mrmd-python"}
        if use_uvx:
            entry["args"] = ["mrmd-python", "--cwd", cwd]
        else:
            entry["args"] = ["--cwd", cwd]
        if venv:
            entry["args"].extend(["--venv", venv])
        print(json.dumps(entry, indent=2))


def main():
    parser = argparse.ArgumentParser(
        prog="mrmd-python",
        description="MCP server for live Python execution. 3 tools: py, py_look, py_ctl.",
    )
    parser.add_argument(
        "command", nargs="?", default=None,
        help="Subcommand: 'install' to show MCP host config",
    )
    parser.add_argument("--http", action="store_true", help="HTTP mode (default: stdio)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--cwd", default=None, help="Working directory (default: current)")
    parser.add_argument("--venv", default=None, help="Virtual environment (default: auto-detect)")
    parser.add_argument("--assets-dir", default=None, help="Asset storage directory")

    args = parser.parse_args()

    # Install subcommand
    if args.command == "install":
        _install(None)
        return

    if args.command and args.command.startswith("install:"):
        _install(args.command.split(":", 1)[1])
        return

    # Resolve cwd and venv
    cwd = args.cwd or os.getcwd()
    venv = args.venv or _find_venv()

    from .mcp_server import create_mcp_server

    mcp = create_mcp_server(cwd=cwd, venv=venv, assets_dir=args.assets_dir)

    if args.http:
        # HTTP mode — FastAPI + MCP
        from .app import create_app
        from .service import RuntimeService
        import uvicorn

        # Reuse the service from the MCP server
        app = create_app(cwd=cwd, venv=venv, assets_dir=args.assets_dir)

        print(f"mrmd-python (HTTP mode)")
        print(f"  MCP:    http://{args.host}:{args.port}/mcp-server/mcp")
        print(f"  Health: http://{args.host}:{args.port}/health")
        print(f"  Assets: http://{args.host}:{args.port}/assets/{{id}}")
        print(f"  cwd:    {cwd}")
        if venv:
            print(f"  venv:   {venv}")
        print()

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    else:
        # STDIO mode — default for MCP hosts
        mcp.run()


if __name__ == "__main__":
    main()
