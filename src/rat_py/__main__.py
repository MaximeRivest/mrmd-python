"""
rat-py: MCP server for live Python execution.

    rat-py              # MCP stdio server (default)
    rat-py --http       # MCP HTTP server
    rat-py install      # register with mcp2cli
"""

import argparse
import os
import shutil
import subprocess
import sys


def _find_venv(cwd: str | None = None) -> str | None:
    """Find the active or nearest venv."""
    if sys.prefix != sys.base_prefix:
        return sys.prefix
    if os.environ.get("VIRTUAL_ENV"):
        return os.environ["VIRTUAL_ENV"]
    search = cwd or os.getcwd()
    for _ in range(6):
        c = os.path.join(search, ".venv")
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "bin", "python")):
            return c
        p = os.path.dirname(search)
        if p == search:
            break
        search = p
    return None


def _server_cmd(venv: str | None) -> str:
    """Best command string for registering with mcp2cli."""
    if venv:
        candidate = os.path.join(venv, "bin", "rat-py")
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("rat-py")
    if found:
        return found
    return "uvx rat-py"


def _cmd_install(args):
    """Register this server with mcp2cli."""
    name = args.name or "py"
    cwd = args.cwd or os.getcwd()
    venv = args.venv or _find_venv(cwd)
    server = _server_cmd(venv)

    mcp2cli = shutil.which("mcp2cli")
    if not mcp2cli:
        print("mcp2cli not found. Install it first:")
        print()
        print("  curl -fsSL https://raw.githubusercontent.com/MaximeRivest/mcp2cli/main/install.sh | sh")
        print()
        return

    # Register
    subprocess.run([mcp2cli, "rm", name], capture_output=True, timeout=5)
    r = subprocess.run(
        [mcp2cli, "add", name, server],
        capture_output=True, timeout=10, text=True,
    )
    if r.returncode != 0:
        print(f"Registration failed: {r.stderr.strip()}")
        return

    print(f"Registered as '{name}'.")
    print()
    print("Usage:")
    print(f"  mcp2cli {name} tools              # list tools")
    print(f"  mcp2cli {name} up                  # start persistent runtime")
    print(f"  {name} run 'x = 42; print(x)'     # run code")
    print(f"  {name} look                        # see variables")
    print(f"  {name} look --at df                # inspect a symbol")
    print(f"  {name} ctl --op reset              # reset namespace")
    print(f"  mcp2cli {name} shell               # interactive mode")
    print(f"  mcp2cli {name} down                # stop runtime")


def main():
    parser = argparse.ArgumentParser(
        prog="rat-py",
        description="rat-py — Run AnyThing: Python. MCP server for live Python execution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
commands:
  (default)  start MCP stdio server
  install    register with mcp2cli

after install:
  mcp2cli py up              # start persistent session
  py run 'x = 42'            # run code
  py look                    # see variables
  py ctl --op reset          # reset
  mcp2cli py down            # stop""",
    )
    parser.add_argument(
        "command", nargs="?", default=None, choices=[None, "install"],
    )
    parser.add_argument("--http", action="store_true", help="run HTTP server instead of stdio")
    parser.add_argument("--port", type=int, default=8717)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cwd", default=None, help="working directory")
    parser.add_argument("--venv", default=None, help="venv path")
    parser.add_argument("--name", default=None, help="mcp2cli server name (default: py)")
    parser.add_argument("--assets-dir", default=None)
    args = parser.parse_args()

    if args.command == "install":
        _cmd_install(args)
        return

    # Run the MCP server
    cwd = args.cwd or os.getcwd()
    venv = args.venv or _find_venv(cwd)

    from .mcp_server import create_mcp_server

    mcp = create_mcp_server(cwd=cwd, venv=venv, assets_dir=args.assets_dir)

    if args.http:
        host = args.host
        port = args.port
        print(f"rat-py (HTTP)")
        print(f"  MCP:    http://{host}:{port}/mcp")
        print(f"  Health: http://{host}:{port}/health")
        print()
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
