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
PORT_START = 8717


def _default_name(cwd: str) -> str:
    """Default runtime name from directory."""
    return Path(cwd).name or "default"


def _state_file(name: str) -> Path:
    return STATE_DIR / f"{name}.json"


def _read_state(name: str) -> dict | None:
    f = _state_file(name)
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return None


def _write_state(name: str, state: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_file(name).write_text(json.dumps(state, indent=2))


def _clear_state(name: str):
    f = _state_file(name)
    if f.exists():
        f.unlink()


def _all_runtimes() -> list[dict]:
    """List all registered runtimes."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    runtimes = []
    for f in sorted(STATE_DIR.glob("*.json")):
        try:
            state = json.loads(f.read_text())
            state["_name"] = f.stem
            state["_alive"] = _is_alive(state.get("pid", 0))
            runtimes.append(state)
        except Exception:
            pass
    return runtimes


def _find_free_port(start: int = PORT_START) -> int:
    """Find a free port starting from start."""
    import socket
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port found in range {start}-{start+99}")


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
    name = args.name or _default_name(cwd)
    venv = args.venv or _find_venv(cwd)

    # Check if already running
    state = _read_state(name)
    if state and _is_alive(state.get("pid", 0)):
        print(f"Runtime '{name}' already running (pid {state['pid']})")
        print(f"  MCP: {state['url']}")
        print(f"\nUse 'mrmd-python stop {name}' to stop it first.")
        return

    # Find port
    port = args.port or _find_free_port()

    # Find python
    python = os.path.join(venv, "bin", "python") if venv else sys.executable

    # Start server in background
    log_dir = Path.home() / ".mrmd" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    host = args.host or "127.0.0.1"
    cmd = [
        python, "-m", "mrmd_python",
        "--http", "--port", str(port), "--host", host, "--cwd", cwd,
    ]
    if venv:
        cmd.extend(["--venv", venv])

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, start_new_session=True, stdout=lf, stderr=subprocess.STDOUT, cwd=cwd,
        )

    # Wait for ready
    url = f"http://{host}:{port}/mcp-server/mcp"
    health_url = f"http://{host}:{port}/health"

    ready = False
    for _ in range(40):
        time.sleep(0.1)
        if not _is_alive(proc.pid):
            print(f"ERROR: Server died. Check {log_file}")
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
    _write_state(name, {
        "pid": proc.pid,
        "port": port,
        "host": host,
        "url": url,
        "name": name,
        "cwd": cwd,
        "venv": venv,
    })

    print(f"Runtime '{name}' started (pid {proc.pid})")
    print(f"  MCP:  {url}")
    print(f"  cwd:  {cwd}")
    if venv:
        print(f"  venv: {venv}")
    print(f"  log:  {log_file}")

    # Auto-register with mcp2cli
    _register_mcp2cli(name, url)

    print(f"\n  Stop: mrmd-python stop {name}")


def _register_mcp2cli(name: str, url: str):
    """Auto-register with mcp2cli if available."""
    if not shutil.which("mcp2cli"):
        print(f"\n  Connect: mcp2cli add {name} --url {url}")
        return

    cli_name = f"python-{name}" if name != "default" else "python"
    try:
        subprocess.run(["mcp2cli", "rm", cli_name], capture_output=True, timeout=5)
        subprocess.run(["mcp2cli", "add", cli_name, "--url", url], capture_output=True, timeout=5)
        result = subprocess.run(
            ["mcp2cli", "expose", cli_name, "--as", cli_name],
            capture_output=True, timeout=5, text=True,
        )
        if result.returncode == 0:
            print(f"\n  mcp2cli: registered as '{cli_name}'")
        else:
            print(f"\n  mcp2cli: registered as '{cli_name}'")
    except Exception:
        pass


def _unregister_mcp2cli(name: str):
    """Remove from mcp2cli."""
    if not shutil.which("mcp2cli"):
        return
    cli_name = f"python-{name}" if name != "default" else "python"
    try:
        subprocess.run(["mcp2cli", "unexpose", cli_name], capture_output=True, timeout=5)
        subprocess.run(["mcp2cli", "rm", cli_name], capture_output=True, timeout=5)
    except Exception:
        pass


def _cmd_stop(args):
    cwd = args.cwd or os.getcwd()
    name = args.name or _default_name(cwd)
    state = _read_state(name)

    if not state:
        print(f"No runtime '{name}' registered.")
        # Show what IS running
        all_rt = _all_runtimes()
        alive = [r for r in all_rt if r["_alive"]]
        if alive:
            print(f"\nRunning runtimes:")
            for r in alive:
                print(f"  {r['_name']}: pid {r.get('pid')} port {r.get('port')} cwd {r.get('cwd', '?')}")
        return

    pid = state.get("pid", 0)
    if _is_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(20):
                if not _is_alive(pid):
                    break
                time.sleep(0.1)
            if _is_alive(pid):
                os.kill(pid, signal.SIGKILL)
        except Exception as e:
            print(f"Warning: {e}")

    _clear_state(name)
    _unregister_mcp2cli(name)
    print(f"Runtime '{name}' stopped (was pid {pid})")


def _cmd_status(args):
    cwd = args.cwd or os.getcwd()
    name = args.name

    if name:
        # Show specific runtime
        _show_runtime_status(name)
        return

    # Show all runtimes
    all_rt = _all_runtimes()
    if not all_rt:
        print("No runtimes registered.")
        print(f"\nStart one: mrmd-python start")
        return

    # Clean up dead ones
    alive = []
    for r in all_rt:
        if r["_alive"]:
            alive.append(r)
        else:
            _clear_state(r["_name"])

    if not alive:
        print("No runtimes running.")
        print(f"\nStart one: mrmd-python start")
        return

    # Table header
    nw = max(len(r["_name"]) for r in alive)
    nw = max(nw, 4)
    print(f"{'NAME':<{nw}}  {'PID':<7}  {'PORT':<5}  {'STATE':<8}  {'EXECS':<5}  CWD")

    for r in alive:
        # Get health
        state_str = "?"
        execs = "?"
        try:
            import urllib.request
            health_url = f"http://{r.get('host', '127.0.0.1')}:{r.get('port')}/health"
            with urllib.request.urlopen(health_url, timeout=1) as resp:
                h = json.loads(resp.read())
                state_str = h.get("state", "?")
                execs = str(h.get("executionCount", 0))
        except Exception:
            pass

        print(f"{r['_name']:<{nw}}  {r.get('pid', '?'):<7}  {r.get('port', '?'):<5}  {state_str:<8}  {execs:<5}  {r.get('cwd', '?')}")


def _show_runtime_status(name: str):
    state = _read_state(name)
    if not state:
        print(f"No runtime '{name}' registered.")
        return

    pid = state.get("pid", 0)
    if not _is_alive(pid):
        _clear_state(name)
        print(f"Runtime '{name}' is DEAD (was pid {pid}, cleaned up)")
        print(f"\nRestart: mrmd-python start --name {name}")
        return

    health_info = ""
    try:
        import urllib.request
        health_url = f"http://{state.get('host', '127.0.0.1')}:{state.get('port')}/health"
        with urllib.request.urlopen(health_url, timeout=2) as r:
            h = json.loads(r.read())
            health_info = f" | {h.get('state', '?')} | {h.get('executionCount', 0)} execs | rev {h.get('stateRevision', 0)}"
    except Exception:
        pass

    print(f"Runtime '{name}' RUNNING (pid {pid}){health_info}")
    print(f"  MCP:  {state.get('url')}")
    print(f"  cwd:  {state.get('cwd')}")
    if state.get("venv"):
        print(f"  venv: {state['venv']}")


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
    parser.add_argument("--name", default=None, help="Runtime name (default: directory name)")
    parser.add_argument("--http", action="store_true", help="HTTP mode (foreground)")
    parser.add_argument("--port", type=int, default=None, help="HTTP port (default: auto)")
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
