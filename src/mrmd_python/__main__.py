"""
mrmd-python: MCP server for live Python execution.

    mrmd-python                   # MCP stdio server
    mrmd-python --http            # MCP HTTP server
    mrmd-python start             # background daemon
    mrmd-python stop              # stop daemon
    mrmd-python status            # show runtimes
    mrmd-python install           # download mcp2cli, register, expose as 'py'
"""

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import stat
import sys
import time
from pathlib import Path

STATE_DIR = Path.home() / ".mrmd" / "runtimes"
PORT_START = 8717


# ── Helpers ──────────────────────────────────────────────────────

def _default_name(cwd: str) -> str:
    return Path(cwd).name or "default"

def _state_file(name: str) -> Path:
    return STATE_DIR / f"{name}.json"

def _read_state(name: str) -> dict | None:
    f = _state_file(name)
    if f.exists():
        try: return json.loads(f.read_text())
        except Exception: pass
    return None

def _write_state(name: str, state: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_file(name).write_text(json.dumps(state, indent=2))

def _clear_state(name: str):
    f = _state_file(name)
    if f.exists(): f.unlink()

def _all_runtimes() -> list[dict]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for f in sorted(STATE_DIR.glob("*.json")):
        try:
            s = json.loads(f.read_text())
            s["_name"] = f.stem
            s["_alive"] = _is_alive(s.get("pid", 0))
            out.append(s)
        except Exception: pass
    return out

def _is_alive(pid: int) -> bool:
    try: os.kill(pid, 0); return True
    except (OSError, ProcessLookupError): return False

def _find_free_port(start: int = PORT_START) -> int:
    import socket
    for port in range(start, start + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError: continue
    raise RuntimeError(f"No free port in {start}-{start+99}")

def _find_venv(cwd: str | None = None) -> str | None:
    if sys.prefix != sys.base_prefix: return sys.prefix
    if os.environ.get("VIRTUAL_ENV"): return os.environ["VIRTUAL_ENV"]
    search = cwd or os.getcwd()
    for _ in range(6):
        c = os.path.join(search, ".venv")
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "bin", "python")):
            return c
        p = os.path.dirname(search)
        if p == search: break
        search = p
    return None

def _server_cmd(venv: str | None) -> str:
    """Find the best command path for registering with mcp2cli."""
    if venv:
        candidate = os.path.join(venv, "bin", "mrmd-python")
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("mrmd-python")
    if found:
        return found
    return "uvx mrmd-python"


# ── mcp2cli management ──────────────────────────────────────────

def _ensure_mcp2cli() -> str | None:
    """Ensure mcp2cli is available. Download if needed. Returns path or None."""
    existing = shutil.which("mcp2cli")
    if existing:
        return existing

    # Download the binary
    system = platform.system().lower()  # linux, darwin, windows
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        arch = "amd64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        print(f"  ⚠ Can't auto-install mcp2cli: unsupported arch {machine}")
        return None

    if system == "windows":
        binary = f"mcp2cli-windows-{arch}.exe"
        dest_name = "mcp2cli.exe"
    else:
        binary = f"mcp2cli-{system}-{arch}"
        dest_name = "mcp2cli"

    url = f"https://github.com/MaximeRivest/mcp2cli/releases/latest/download/{binary}"
    install_dir = Path.home() / ".local" / "bin"
    install_dir.mkdir(parents=True, exist_ok=True)
    dest = install_dir / dest_name

    print(f"  Downloading mcp2cli...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  ✓ Installed to {dest}")

        # Check if ~/.local/bin is on PATH
        if str(install_dir) not in os.environ.get("PATH", ""):
            print(f"  ⚠ Add {install_dir} to your PATH:")
            print(f'    export PATH="{install_dir}:$PATH"')

        return str(dest)
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        print(f"  Manual install: curl -fsSL https://raw.githubusercontent.com/MaximeRivest/mcp2cli/main/install.sh | sh")
        return None


# ── Commands ─────────────────────────────────────────────────────

def _cmd_install(args):
    cwd = args.cwd or os.getcwd()
    venv = args.venv or _find_venv(cwd)
    name = args.name or "py"
    server = _server_cmd(venv)

    print("mrmd-python install\n")

    # 1. Ensure mcp2cli
    mcp2cli = _ensure_mcp2cli()
    if not mcp2cli:
        return

    # 2. Register
    print(f"  Registering as '{name}'...")
    subprocess.run([mcp2cli, "rm", name], capture_output=True, timeout=5)
    r = subprocess.run([mcp2cli, "add", name, server],
                       capture_output=True, timeout=10, text=True)
    if r.returncode != 0:
        print(f"  ✗ Registration failed: {r.stderr.strip()}")
        return
    print(f"  ✓ Registered")

    # 3. Done
    print(f"""
Done! Usage:

  mcp2cli {name} up                    # start persistent runtime
  {name} run 'x = 42; print(x)'       # run code
  {name} look                          # see variables
  {name} look --at df                  # inspect a symbol
  {name} ctl --op reset               # reset namespace
  mcp2cli {name} down                  # stop runtime

Shared mode (multiple clients, one runtime):
  mrmd-python start
""")


def _cmd_start(args):
    cwd = args.cwd or os.getcwd()
    name = args.name or _default_name(cwd)
    venv = args.venv or _find_venv(cwd)

    state = _read_state(name)
    if state and _is_alive(state.get("pid", 0)):
        print(f"Runtime '{name}' already running (pid {state['pid']})")
        print(f"  MCP: {state['url']}")
        return

    port = args.port or _find_free_port()
    python = os.path.join(venv, "bin", "python") if venv else sys.executable
    host = args.host or "127.0.0.1"

    log_dir = Path.home() / ".mrmd" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name}.log"

    cmd = [python, "-m", "mrmd_python", "--http",
           "--port", str(port), "--host", host, "--cwd", cwd]
    if venv: cmd.extend(["--venv", venv])

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, start_new_session=True,
                                stdout=lf, stderr=subprocess.STDOUT, cwd=cwd)

    url = f"http://{host}:{port}/mcp"
    for _ in range(60):
        time.sleep(0.1)
        if not _is_alive(proc.pid):
            print(f"ERROR: died. Check {log_file}"); return
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=1) as r:
                if r.status == 200: break
        except Exception: pass
    else:
        try: os.kill(proc.pid, signal.SIGTERM)
        except Exception: pass
        print(f"ERROR: timeout. Check {log_file}"); return

    _write_state(name, {"pid": proc.pid, "port": port, "host": host,
                         "url": url, "name": name, "cwd": cwd, "venv": venv})

    print(f"Runtime '{name}' started (pid {proc.pid})")
    print(f"  MCP:  {url}")
    print(f"  cwd:  {cwd}")
    if venv: print(f"  venv: {venv}")
    print(f"  log:  {log_file}")

    # Auto-register with mcp2cli if available
    cli_name = f"py-{name}" if name not in ("py", "default") else "py"
    mcp2cli = shutil.which("mcp2cli")
    if mcp2cli:
        subprocess.run([mcp2cli, "rm", cli_name], capture_output=True, timeout=5)
        subprocess.run([mcp2cli, "add", cli_name, "--url", url],
                       capture_output=True, timeout=5)
        print(f"\n  mcp2cli: registered as '{cli_name}'")
        print(f"  Try: {cli_name} run 'print(\"hello\")'")

    print(f"\n  Stop: mrmd-python stop --name {name}")


def _cmd_stop(args):
    cwd = args.cwd or os.getcwd()
    name = args.name or _default_name(cwd)
    state = _read_state(name)
    if not state:
        print(f"No runtime '{name}'.")
        alive = [r for r in _all_runtimes() if r["_alive"]]
        if alive: print("Running:", ", ".join(r["_name"] for r in alive))
        return
    pid = state.get("pid", 0)
    if _is_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(20):
                if not _is_alive(pid): break
                time.sleep(0.1)
            if _is_alive(pid): os.kill(pid, signal.SIGKILL)
        except Exception: pass
    _clear_state(name)
    # Clean mcp2cli
    cli_name = f"py-{name}" if name not in ("py", "default") else "py"
    mcp2cli = shutil.which("mcp2cli")
    if mcp2cli:
        subprocess.run([mcp2cli, "rm", cli_name], capture_output=True, timeout=5)
    print(f"Runtime '{name}' stopped (was pid {pid})")


def _cmd_status(args):
    rts = _all_runtimes()
    alive = [r for r in rts if r["_alive"]]
    for r in rts:
        if not r["_alive"]: _clear_state(r["_name"])
    if not alive:
        print("No runtimes running.")
        return
    nw = max(len(r["_name"]) for r in alive)
    print(f"{'NAME':<{max(nw,4)}}  {'PID':<7}  {'PORT':<5}  {'STATE':<6}  CWD")
    for r in alive:
        st = "?"
        try:
            import urllib.request
            with urllib.request.urlopen(
                f"http://{r.get('host','127.0.0.1')}:{r.get('port')}/health", timeout=1
            ) as resp: st = json.loads(resp.read()).get("state", "?")
        except Exception: pass
        print(f"{r['_name']:<{max(nw,4)}}  {r.get('pid','?'):<7}  {r.get('port','?'):<5}  {st:<6}  {r.get('cwd','?')}")


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="mrmd-python",
        description="MCP server for live Python execution. Tools: run, look, ctl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""commands:
  (none)     MCP stdio server (for Claude Desktop, Cursor)
  install    download mcp2cli, register as 'py', ready to use
  start      shared background daemon (multi-client)
  stop       stop daemon
  status     show running runtimes

after install:
  mcp2cli py up              # start persistent session
  py run 'x = 42'            # run code
  py look                    # see variables
  py ctl --op reset          # reset""")
    parser.add_argument("command", nargs="?", default=None,
                        choices=[None, "start", "stop", "status", "install"])
    parser.add_argument("--name", default=None)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--venv", default=None)
    parser.add_argument("--assets-dir", default=None)
    args = parser.parse_args()

    if args.command == "install": _cmd_install(args); return
    if args.command == "start": _cmd_start(args); return
    if args.command == "stop": _cmd_stop(args); return
    if args.command == "status": _cmd_status(args); return

    cwd = args.cwd or os.getcwd()
    venv = args.venv or _find_venv(cwd)

    from .mcp_server import create_mcp_server
    mcp = create_mcp_server(cwd=cwd, venv=venv, assets_dir=args.assets_dir)

    if args.http:
        port = args.port or PORT_START
        host = args.host or "127.0.0.1"
        print(f"mrmd-python (HTTP)\n  MCP: http://{host}:{port}/mcp\n  Health: http://{host}:{port}/health\n")
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
