"""
Independent Runtime Daemon

This module implements a truly independent Python runtime that:
- Runs as a daemon process (survives parent death)
- Exposes MRP endpoints via HTTP
- Registers itself in ~/.mrmd/runtimes/ for discovery
- Can be killed independently via its PID

Architecture:
    ~/.mrmd/
    ├── runtimes/
    │   └── {id}.json       # Runtime registry (PID, port, etc.)
    └── logs/
        └── {id}.log        # Runtime logs

Usage:
    # Spawn a new daemon runtime
    python -m mrmd_python.runtime_daemon --id myruntime --port 0 --venv /path/to/venv

    # The daemon will:
    # 1. Double-fork to detach from parent
    # 2. Write its info to ~/.mrmd/runtimes/{id}.json
    # 3. Start HTTP server on assigned port
    # 4. Keep running until explicitly killed
"""

import os
import sys
import json
import signal
import atexit
import socket
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Runtime registry directory
MRMD_DIR = Path.home() / ".mrmd"
RUNTIMES_DIR = MRMD_DIR / "runtimes"
LOGS_DIR = MRMD_DIR / "logs"


def get_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


def ensure_dirs():
    """Ensure runtime directories exist."""
    RUNTIMES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_runtime_info_path(runtime_id: str) -> Path:
    """Get path to runtime info file."""
    return RUNTIMES_DIR / f"{runtime_id}.json"


def get_runtime_log_path(runtime_id: str) -> Path:
    """Get path to runtime log file."""
    return LOGS_DIR / f"{runtime_id}.log"


def read_runtime_info(runtime_id: str) -> Optional[dict]:
    """Read runtime info from registry."""
    path = get_runtime_info_path(runtime_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def write_runtime_info(info: dict):
    """Write runtime info to registry."""
    ensure_dirs()
    path = get_runtime_info_path(info["id"])
    path.write_text(json.dumps(info, indent=2))


def remove_runtime_info(runtime_id: str):
    """Remove runtime info from registry."""
    path = get_runtime_info_path(runtime_id)
    if path.exists():
        path.unlink()


def list_runtimes() -> list[dict]:
    """List all registered runtimes."""
    ensure_dirs()
    runtimes = []
    for path in RUNTIMES_DIR.glob("*.json"):
        try:
            info = json.loads(path.read_text())
            # Check if process is still alive
            pid = info.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)  # Signal 0 = check if alive
                    info["alive"] = True
                except OSError:
                    info["alive"] = False
            runtimes.append(info)
        except Exception:
            pass
    return runtimes


def is_runtime_alive(runtime_id: str) -> bool:
    """Check if a runtime is still running."""
    info = read_runtime_info(runtime_id)
    if not info:
        return False
    pid = info.get("pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill_runtime(runtime_id: str) -> bool:
    """
    Kill a runtime by ID.

    Uses process group kill to ensure all child processes are terminated.
    This is critical for GPU workloads (vLLM, etc.) where child processes
    hold CUDA contexts that must be released.
    """
    info = read_runtime_info(runtime_id)
    if not info:
        return False
    pid = info.get("pid")
    if not pid:
        return False
    try:
        # Get the process group ID
        # The daemon becomes a session leader via setsid(), so its PGID
        # is its own PID. Child processes inherit this PGID.
        try:
            pgid = os.getpgid(pid)
        except OSError:
            # Process already dead, just clean up
            remove_runtime_info(runtime_id)
            return True

        # Kill the entire process group - this catches:
        # - The main runtime process
        # - vLLM worker threads/processes
        # - Any other child processes holding GPU memory
        os.killpg(pgid, signal.SIGKILL)
        remove_runtime_info(runtime_id)
        return True
    except OSError:
        # Process group might already be dead
        remove_runtime_info(runtime_id)
        return False


def daemonize():
    """
    Double-fork to create a daemon process.

    This ensures the process is fully detached from the parent:
    - First fork: Creates child, parent exits
    - setsid(): Child becomes session leader
    - Second fork: Grandchild can't acquire controlling terminal

    NOTE: We use os._exit(0) instead of sys.exit(0) because:
    - sys.exit() raises SystemExit which can propagate through ASGI/FastAPI
    - os._exit() terminates immediately without raising exceptions
    - This is critical when spawn_daemon is called from async context
    """
    # First fork
    pid = os.fork()
    if pid > 0:
        # Parent exits - use os._exit to avoid SystemExit propagation
        os._exit(0)

    # Child becomes session leader
    os.setsid()

    # Second fork
    pid = os.fork()
    if pid > 0:
        # First child exits - use os._exit to avoid SystemExit propagation
        os._exit(0)

    # Grandchild continues as daemon
    # Make ourselves a process group leader so our PID == PGID
    # This ensures killpg(pid, SIGKILL) kills us and all our children
    try:
        os.setpgid(0, 0)
    except OSError:
        pass  # May fail if already a process group leader

    # Change working directory to root to avoid holding mount points
    # (We'll change to cwd later if specified)
    os.chdir("/")

    # Reset file creation mask
    os.umask(0)

    # Close standard file descriptors
    sys.stdin.close()
    sys.stdout.close()
    sys.stderr.close()

    # Redirect to /dev/null
    sys.stdin = open('/dev/null', 'r')
    sys.stdout = open('/dev/null', 'w')
    sys.stderr = open('/dev/null', 'w')


def run_daemon(
    runtime_id: str,
    port: int,
    venv: str,
    cwd: Optional[str] = None,
    assets_dir: Optional[str] = None,
):
    """
    Run the daemon runtime server.

    This is called after daemonize() in the grandchild process.
    """
    # Redirect output to log file
    ensure_dirs()
    log_path = get_runtime_log_path(runtime_id)
    log_file = open(log_path, 'a')
    sys.stdout = log_file
    sys.stderr = log_file

    # Change to working directory
    if cwd:
        os.chdir(cwd)

    # Get actual port if 0 was specified
    if port == 0:
        port = get_free_port()

    # Write runtime info
    info = {
        "id": runtime_id,
        "pid": os.getpid(),
        "port": port,
        "host": "localhost",
        "url": f"http://localhost:{port}/mrp/v1",
        "venv": venv,
        "cwd": cwd or os.getcwd(),
        "assets_dir": assets_dir,
        "created": datetime.now(timezone.utc).isoformat(),
        "lastActivity": datetime.now(timezone.utc).isoformat(),
    }
    write_runtime_info(info)

    # Register cleanup on exit
    def cleanup():
        remove_runtime_info(runtime_id)
    atexit.register(cleanup)

    # Handle SIGTERM gracefully
    def handle_sigterm(signum, frame):
        cleanup()
        sys.exit(0)
    signal.signal(signal.SIGTERM, handle_sigterm)

    print(f"[{datetime.now().isoformat()}] Starting runtime daemon {runtime_id} on port {port}")
    print(f"  PID: {os.getpid()}")
    print(f"  venv (param): {venv}")
    print(f"  sys.prefix (actual Python): {sys.prefix}")
    print(f"  sys.executable: {sys.executable}")
    print(f"  cwd: {cwd}")
    print(f"  VENV MISMATCH: {venv != sys.prefix}")
    sys.stdout.flush()

    # Import and run the server
    # We import here to avoid loading heavy deps before daemonizing
    import uvicorn
    from .server import create_app

    print(f"  Calling create_app with venv={venv}, daemon_mode=True")
    sys.stdout.flush()

    # CRITICAL: daemon_mode=True tells the server to use local IPythonWorker
    # instead of spawning more daemons (which would cause infinite recursion)
    app = create_app(cwd=cwd, assets_dir=assets_dir, venv=venv, daemon_mode=True)
    print(f"  App created successfully")
    sys.stdout.flush()

    # Run uvicorn (this blocks)
    uvicorn.run(
        app,
        host="localhost",
        port=port,
        log_level="info",
        access_log=False,
    )


def spawn_daemon(
    runtime_id: str,
    port: int = 0,
    venv: Optional[str] = None,
    cwd: Optional[str] = None,
    assets_dir: Optional[str] = None,
) -> dict:
    """
    Spawn a new daemon runtime and return its info.

    This function:
    1. Forks a daemon process
    2. The daemon writes its info to ~/.mrmd/runtimes/{id}.json
    3. Returns the runtime info once available
    """
    # Check if already running
    if is_runtime_alive(runtime_id):
        return read_runtime_info(runtime_id)

    # Determine venv
    if not venv:
        if sys.prefix != sys.base_prefix:
            venv = sys.prefix
        else:
            venv = os.environ.get("VIRTUAL_ENV", sys.prefix)

    # Determine cwd
    if not cwd:
        cwd = os.getcwd()

    # Get port
    if port == 0:
        port = get_free_port()

    # Fork the daemon
    pid = os.fork()

    if pid == 0:
        # Child process - daemonize and run
        daemonize()
        run_daemon(runtime_id, port, venv, cwd, assets_dir)
        sys.exit(0)
    else:
        # Parent process - wait for runtime to be ready
        # The daemon will write its info file when ready
        info_path = get_runtime_info_path(runtime_id)

        # Wait up to 10 seconds for the daemon to start
        for _ in range(100):
            if info_path.exists():
                info = read_runtime_info(runtime_id)
                if info and is_runtime_alive(runtime_id):
                    return info
            time.sleep(0.1)

        raise RuntimeError(f"Daemon {runtime_id} failed to start")


def main():
    """CLI entry point for runtime daemon."""
    parser = argparse.ArgumentParser(description="MRMD Python Runtime Daemon")
    parser.add_argument("--id", help="Runtime ID (required for starting a daemon)")
    parser.add_argument("--port", type=int, default=0, help="Port (0 for auto)")
    parser.add_argument("--venv", help="Virtual environment path")
    parser.add_argument("--cwd", help="Working directory")
    parser.add_argument("--assets-dir", help="Assets directory")
    parser.add_argument("--foreground", action="store_true", help="Run in foreground (no daemon)")

    # Management commands
    parser.add_argument("--list", action="store_true", help="List all runtimes")
    parser.add_argument("--kill", metavar="ID", help="Kill a runtime by ID")
    parser.add_argument("--info", metavar="ID", help="Get info about a runtime")
    parser.add_argument("--kill-all", action="store_true", help="Kill all runtimes")

    args = parser.parse_args()

    # Handle management commands
    if args.list:
        runtimes = list_runtimes()
        if not runtimes:
            print("No runtimes registered")
        else:
            for rt in runtimes:
                status = "ALIVE" if rt.get("alive") else "DEAD"
                print(f"{rt['id']}: {status} pid={rt.get('pid')} port={rt.get('port')} url={rt.get('url')}")
        return

    if args.kill:
        if kill_runtime(args.kill):
            print(f"Killed runtime {args.kill}")
        else:
            print(f"Runtime {args.kill} not found or already dead")
        return

    if args.info:
        info = read_runtime_info(args.info)
        if info:
            info["alive"] = is_runtime_alive(args.info)
            print(json.dumps(info, indent=2))
        else:
            print(f"Runtime {args.info} not found")
        return

    if args.kill_all:
        runtimes = list_runtimes()
        killed = 0
        for rt in runtimes:
            if kill_runtime(rt["id"]):
                print(f"Killed {rt['id']}")
                killed += 1
        print(f"Killed {killed} runtime(s)")
        return

    # Starting a daemon requires --id
    if not args.id:
        parser.error("--id is required to start a daemon runtime")

    # Determine venv
    venv = args.venv
    if not venv:
        if sys.prefix != sys.base_prefix:
            venv = sys.prefix
        else:
            venv = os.environ.get("VIRTUAL_ENV", sys.prefix)

    if args.foreground:
        # Run in foreground (for debugging)
        run_daemon(args.id, args.port or 0, venv, args.cwd, args.assets_dir)
    else:
        # Daemonize
        info = spawn_daemon(args.id, args.port or 0, venv, args.cwd, args.assets_dir)
        print(f"Started daemon runtime:")
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
