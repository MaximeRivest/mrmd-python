"""
mrmd-python: MCP server for live Python execution.

    mrmd-python              # MCP over stdio (for MCP hosts)
    mrmd-python start        # start shared runtime server (background)
    mrmd-python stop         # stop it
    mrmd-python status       # show if running
    mrmd-python install      # show config for Claude Desktop / Cursor / mcp2cli
    mrmd-python --http       # start HTTP server (foreground)
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── State file ───────────────────────────────────────────────────

STATE_DIR = Path.home() / ".mrmd" / "runtimes"
DEFAULT_PORT = 8717


def _state_file(cwd: str) -> Path:
    """State file path for a given cwd."""
    # Use a hash of cwd to allow multiple runtimes
    import hashlib
    h = hashlib.sha256(cwd.encode()).hexdigest()[:12]
    return STATE_DIR / f"{h}.json"


def _read_state(cwd: str) -> dict | None:
    f = _state_file(cwd)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return None


def _write_state(cwd: str, state: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_file(cwd).write_text(json.dumps(state, indent=2))


def _clear_state(cwd: str):
    f = _state_file(cwd)
    if f.exists():
        f.unlink()


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ── Venv detection ───────────────────────────────────────────────

def _find_venv(cwd: str | None = None) -> str | None:
    if sys.prefix != sys.base_prefix:
        return sys.prefix
    if os.environ.get("VIRTUAL_ENV"):
        return os.environ["VIRTUAL_ENV"]
    search = cwd or os.getcwd()
    for _ in range(6):
        candidate = os.path.join(search, ".venv")
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "bin", "python")):
            return candidate
        parent = os.path.dirname(search)
        if parent == search:
            break
        search = parent
    return None


# ── Commands ─────────────────────────────────────────────────────

def _cmd_start(args):
    cwd = args.cwd or os.getcwd()
    port = args.port or DEFAULT_PORT
    venv = args.venv or _find_venv(cwd)

    # Check if already running
    state = _read_state(cwd)
    if state and _is_alive(state.get("pid", 0)):
        url = state["url"]
        print(f"Runtime already running (pid {state['pid']})")
        print(f"  URL: {url}")
        print(f"\nUse 'mrmd-python stop' to stop it first.")
        return

    # Find python with mrmd-python installed
    if venv:
        python = os.path.join(venv, "bin", "python")
    else:
        python = sys.executable

    # Start server in background
    log_dir = Path.home() / ".mrmd" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "runtime.log"

    cmd = [
        python, "-m", "mrmd_python",
        "--http",
        "--port", str(port),
        "--host", args.host or "127.0.0.1",
        "--cwd", cwd,
    ]
    if venv:
        cmd.extend(["--venv", venv])

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=cwd,
        )

    # Wait for server to be ready
    host = args.host or "127.0.0.1"
    url = f"http://{host}:{port}/mcp-server/mcp"
    health_url = f"http://{host}:{port}/health"

    ready = False
    for _ in range(40):  # 4 seconds max
        time.sleep(0.1)
        if not _is_alive(proc.pid):
            print(f"ERROR: Server process died. Check {log_file}")
            return
        try:
            import urllib.request
            with urllib.request.urlopen(health_url, timeout=1) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            pass

    if not ready:
        print(f"ERROR: Server didn't start within 4s. Check {log_file}")
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        return

    # Save state
    _write_state(cwd, {
        "pid": proc.pid,
        "port": port,
        "host": host,
        "url": url,
        "cwd": cwd,
        "venv": venv,
    })

    print(f"Runtime started (pid {proc.pid})")
    print(f"  MCP: {url}")
    print(f"  cwd: {cwd}")
    if venv:
        print(f"  venv: {venv}")
    print(f"  log: {log_file}")

    # Auto-register with mcp2cli if available
    if shutil.which("mcp2cli"):
        try:
            subprocess.run(
                ["mcp2cli", "rm", "python"],
                capture_output=True, timeout=5,
            )
            subprocess.run(
                ["mcp2cli", "add", "python", "--url", url],
                capture_output=True, timeout=5,
            )
            # Try to expose
            result = subprocess.run(
                ["mcp2cli", "expose", "python", "--as", "py"],
                capture_output=True, timeout=5, text=True,
            )
            if result.returncode == 0:
                print(f"\n  mcp2cli registered and exposed as 'py'")
                print(f"  Try: py py \"print('hello')\"")
            else:
                print(f"\n  mcp2cli registered as 'python'")
                print(f"  Try: mcp2cli tool python py --code \"print('hello')\"")
        except Exception:
            pass
    else:
        print(f"\n  Connect: mcp2cli add python --url {url}")

    print(f"\n  Stop: mrmd-python stop")


def _cmd_stop(args):
    cwd = args.cwd or os.getcwd()
    state = _read_state(cwd)

    if not state:
        print("No runtime registered for this directory.")
        return

    pid = state.get("pid", 0)
    if _is_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for graceful shutdown
            for _ in range(20):
                if not _is_alive(pid):
                    break
                time.sleep(0.1)
            if _is_alive(pid):
                os.kill(pid, signal.SIGKILL)
        except Exception as e:
            print(f"Warning: {e}")

    _clear_state(cwd)

    # Clean up mcp2cli
    if shutil.which("mcp2cli"):
        try:
            subprocess.run(["mcp2cli", "unexpose", "python"], capture_output=True, timeout=5)
            subprocess.run(["mcp2cli", "rm", "python"], capture_output=True, timeout=5)
        except Exception:
            pass

    print(f"Runtime stopped (was pid {pid})")


def _cmd_status(args):
    cwd = args.cwd or os.getcwd()
    state = _read_state(cwd)

    if not state:
        print("No runtime registered for this directory.")
        print(f"\nStart one: mrmd-python start")
        return

    pid = state.get("pid", 0)
    alive = _is_alive(pid)

    if alive:
        # Try to get health
        health_info = ""
        try:
            import urllib.request
            url = state["url"].replace("/mcp-server/mcp", "/health")
            with urllib.request.urlopen(url, timeout=2) as r:
                h = json.loads(r.read())
                health_info = f" | {h.get('state', '?')} | {h.get('executionCount', 0)} execs | rev {h.get('stateRevision', 0)}"
        except Exception:
            pass

        print(f"RUNNING (pid {pid}){health_info}")
        print(f"  MCP: {state.get('url')}")
        print(f"  cwd: {state.get('cwd')}")
        if state.get("venv"):
            print(f"  venv: {state['venv']}")
    else:
        _clear_state(cwd)
        print(f"DEAD (was pid {pid}, cleaned up)")
        print(f"\nRestart: mrmd-python start")


def _cmd_install(args):
    cwd = args.cwd or os.getcwd()
    venv = args.venv or _find_venv(cwd)
    state = _read_state(cwd)

    print("# mrmd-python MCP configuration\n")

    # If a runtime is running, show shared URL config
    if state and _is_alive(state.get("pid", 0)):
        url = state["url"]
        print(f"## Runtime is running at {url}\n")

        print("## Claude Desktop / Cursor / any MCP client")
        print(json.dumps({"mcpServers": {"python": {"url": url}}}, indent=2))
        print()

        print("## mcp2cli")
        print(f"mcp2cli add python --url {url}")
        print()
        return

    # No runtime running — show stdio config
    cmd = "mrmd-python"
    use_uvx = not shutil.which("mrmd-python")

    print("## Stdio mode (each host gets its own runtime)\n")

    print("## Claude Desktop")
    entry = {"command": "uvx" if use_uvx else cmd}
    a = ["mrmd-python"] if use_uvx else []
    a.extend(["--cwd", cwd])
    if venv:
        a.extend(["--venv", venv])
    entry["args"] = a
    print(json.dumps({"mcpServers": {"python": entry}}, indent=2))
    print()

    print("## Shared mode (all clients share one runtime)\n")
    print("# Start runtime:")
    print(f"mrmd-python start --cwd {cwd}")
    print("# Then connect any client to the printed URL")
    print()


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="mrmd-python",
        description="MCP server for live Python execution. 3 tools: py, py_look, py_ctl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""commands:
  (none)     start MCP server on stdio (for MCP hosts)
  start      start shared runtime server (background HTTP)
  stop       stop the shared runtime server
  status     show runtime status
  install    show config for Claude Desktop / Cursor / mcp2cli""",
    )
    parser.add_argument(
        "command", nargs="?", default=None,
        choices=[None, "start", "stop", "status", "install"],
        help="Subcommand",
    )
    parser.add_argument("--http", action="store_true", help="HTTP mode (foreground)")
    parser.add_argument("--port", type=int, default=None, help=f"HTTP port (default: {DEFAULT_PORT})")
    parser.add_argument("--host", default=None, help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--cwd", default=None, help="Working directory (default: current)")
    parser.add_argument("--venv", default=None, help="Virtual environment (default: auto-detect)")
    parser.add_argument("--assets-dir", default=None, help="Asset storage directory")

    args = parser.parse_args()

    if args.command == "start":
        _cmd_start(args)
        return

    if args.command == "stop":
        _cmd_stop(args)
        return

    if args.command == "status":
        _cmd_status(args)
        return

    if args.command == "install":
        _cmd_install(args)
        return

    # Resolve cwd and venv
    cwd = args.cwd or os.getcwd()
    venv = args.venv or _find_venv(cwd)

    from .mcp_server import create_mcp_server

    mcp = create_mcp_server(cwd=cwd, venv=venv, assets_dir=args.assets_dir)

    if args.http:
        from .app import create_app
        import uvicorn

        port = args.port or DEFAULT_PORT
        host = args.host or "127.0.0.1"
        app = create_app(cwd=cwd, venv=venv, assets_dir=args.assets_dir)

        print(f"mrmd-python (HTTP)")
        print(f"  MCP:    http://{host}:{port}/mcp-server/mcp")
        print(f"  Health: http://{host}:{port}/health")
        print(f"  cwd:    {cwd}")
        if venv:
            print(f"  venv:   {venv}")
        print()

        uvicorn.run(app, host=host, port=port, log_level="info")
    else:
        # Stdio — default for MCP hosts
        mcp.run()


if __name__ == "__main__":
    main()
