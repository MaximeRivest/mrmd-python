"""
mrmd-python CLI

Independent Python runtime for MRMD notebooks.

Usage:
    # Start a daemon runtime (default)
    mrmd-python
    mrmd-python --id my-session --venv /path/to/venv

    # Management commands
    mrmd-python --list              # List running runtimes
    mrmd-python --kill my-session   # Kill a runtime
    mrmd-python --kill-all          # Kill all runtimes
    mrmd-python --info my-session   # Get runtime info

    # Development/debugging
    mrmd-python --foreground        # Run in foreground (no daemon)

    # Using with uvx (no install needed)
    uvx mrmd-python
    uvx mrmd-python --list
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="mrmd-python",
        description="Independent Python runtime for MRMD notebooks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mrmd-python                     Start default runtime (daemon mode)
  mrmd-python --id vllm           Start runtime named 'vllm'
  mrmd-python --venv ~/.venv/ml   Use specific virtual environment
  mrmd-python --list              List all running runtimes
  mrmd-python --kill vllm         Kill the 'vllm' runtime
  mrmd-python --foreground        Run in foreground (for debugging)

Virtual Environment Detection:
  If --venv is not specified, mrmd-python auto-detects:
  1. Current venv (if running inside one)
  2. VIRTUAL_ENV environment variable
  3. Falls back to system Python

Using with uvx (no install needed):
  uvx mrmd-python                 Start default runtime
  uvx mrmd-python --list          List runtimes
""",
    )

    # Runtime configuration
    parser.add_argument(
        "--id",
        default="default",
        help="Runtime ID (default: 'default')",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to bind to (0 = auto-assign)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost, use 0.0.0.0 for containers)",
    )
    parser.add_argument(
        "--venv",
        default=None,
        help="Virtual environment path (auto-detected if not specified)",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory (default: current directory)",
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Directory for saving assets like plots",
    )

    # Execution mode
    parser.add_argument(
        "--foreground",
        action="store_true",
        help="Run in foreground instead of daemon mode",
    )

    # Management commands
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all running runtimes",
    )
    parser.add_argument(
        "--kill",
        metavar="ID",
        help="Kill a runtime by ID",
    )
    parser.add_argument(
        "--kill-all",
        action="store_true",
        help="Kill all running runtimes",
    )
    parser.add_argument(
        "--info",
        metavar="ID",
        help="Get detailed info about a runtime",
    )

    args = parser.parse_args()

    # Import daemon functions (lazy import for fast --help)
    from .runtime_daemon import (
        list_runtimes,
        kill_runtime,
        read_runtime_info,
        is_runtime_alive,
        spawn_daemon,
        run_daemon,
    )

    # Handle management commands
    if args.list:
        runtimes = list_runtimes()
        if not runtimes:
            print("No runtimes running")
            print("\nStart one with: mrmd-python")
        else:
            print(f"{'ID':<20} {'STATUS':<8} {'PID':<10} {'PORT':<8} {'URL'}")
            print("-" * 80)
            for rt in runtimes:
                status = "ALIVE" if rt.get("alive") else "DEAD"
                print(f"{rt['id']:<20} {status:<8} {rt.get('pid', '-'):<10} {rt.get('port', '-'):<8} {rt.get('url', '-')}")
        return

    if args.kill:
        if kill_runtime(args.kill):
            print(f"Killed runtime '{args.kill}'")
        else:
            print(f"Runtime '{args.kill}' not found or already dead")
        return

    if args.kill_all:
        runtimes = list_runtimes()
        if not runtimes:
            print("No runtimes to kill")
            return
        killed = 0
        for rt in runtimes:
            if kill_runtime(rt["id"]):
                print(f"Killed '{rt['id']}'")
                killed += 1
        print(f"\nKilled {killed} runtime(s)")
        return

    if args.info:
        info = read_runtime_info(args.info)
        if info:
            info["alive"] = is_runtime_alive(args.info)
            print(json.dumps(info, indent=2))
        else:
            print(f"Runtime '{args.info}' not found")
        return

    # Determine venv
    venv = args.venv
    if not venv:
        # Auto-detect venv
        if sys.prefix != sys.base_prefix:
            venv = sys.prefix
        else:
            venv = os.environ.get("VIRTUAL_ENV", sys.prefix)

    # Determine cwd
    cwd = args.cwd or os.getcwd()

    # Check if runtime already exists
    existing = read_runtime_info(args.id)
    if existing and is_runtime_alive(args.id):
        print(f"Runtime '{args.id}' is already running:")
        print(f"  URL:  {existing.get('url')}")
        print(f"  PID:  {existing.get('pid')}")
        print(f"  venv: {existing.get('venv')}")
        print(f"\nTo kill it: mrmd-python --kill {args.id}")
        return

    if args.foreground:
        # Run in foreground (blocking, for debugging)
        print(f"Starting mrmd-python in foreground mode...")
        print(f"  ID:   {args.id}")
        print(f"  venv: {venv}")
        print(f"  cwd:  {cwd}")
        print(f"  port: {args.port or 'auto'}")
        print(f"\nPress Ctrl+C to stop\n")
        run_daemon(args.id, args.port, venv, cwd, args.assets_dir, host=args.host)
    else:
        # Spawn daemon (non-blocking)
        print(f"Starting mrmd-python daemon...")
        info = spawn_daemon(
            runtime_id=args.id,
            port=args.port,
            venv=venv,
            cwd=cwd,
            assets_dir=args.assets_dir,
        )
        print(f"""
Runtime '{info['id']}' started:
  URL:  {info['url']}
  PID:  {info['pid']}
  venv: {info['venv']}
  cwd:  {info['cwd']}

Management:
  mrmd-python --list          List all runtimes
  mrmd-python --info {info['id']:<8} Get runtime info
  mrmd-python --kill {info['id']:<8} Kill this runtime
""")


if __name__ == "__main__":
    main()
