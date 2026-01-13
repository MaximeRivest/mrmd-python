"""
mrmd-python CLI

Usage:
    mrmd-python [--host HOST] [--port PORT] [--cwd DIR] [--assets-dir DIR]
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Python runtime server implementing the MRMD Runtime Protocol (MRP)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory (default: current directory)",
    )
    parser.add_argument(
        "--assets-dir",
        default=None,
        help="Directory for saving assets (default: .mrmd-assets in cwd)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--venv",
        default=None,
        help="Path to virtual environment to use for code execution",
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    import uvicorn
    from .server import create_app

    cwd = args.cwd or os.getcwd()
    assets_dir = args.assets_dir or os.path.join(cwd, ".mrmd-assets")
    venv = args.venv

    venv_display = venv[:43] if venv else "System Python"
    print(
        f"""
╔═══════════════════════════════════════════════════════════════╗
║                    mrmd-python MRP Server                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Running on: http://{args.host}:{args.port}/mrp/v1{' ' * (26 - len(args.host) - len(str(args.port)))}║
║                                                               ║
║  Endpoints:                                                   ║
║    GET  /mrp/v1/capabilities                                  ║
║    GET  /mrp/v1/sessions                                      ║
║    POST /mrp/v1/execute                                       ║
║    POST /mrp/v1/execute/stream                                ║
║    POST /mrp/v1/complete                                      ║
║    POST /mrp/v1/inspect                                       ║
║    POST /mrp/v1/hover                                         ║
║    POST /mrp/v1/variables                                     ║
║                                                               ║
║  Working directory: {cwd[:43]:<43}║
║  Assets directory:  {assets_dir[:43]:<43}║
║  Virtual env:       {venv_display:<43}║
║                                                               ║
║  Press Ctrl+C to stop                                         ║
╚═══════════════════════════════════════════════════════════════╝
"""
    )

    # Create app factory for uvicorn
    def app_factory():
        return create_app(cwd=cwd, assets_dir=assets_dir, venv=venv)

    if args.reload:
        global _cwd, _assets_dir, _venv
        _cwd = cwd
        _assets_dir = assets_dir
        _venv = venv
        uvicorn.run(
            "mrmd_python.cli:_create_app_for_reload",
            host=args.host,
            port=args.port,
            reload=True,
            access_log=False,  # Disable access log to prevent it leaking into execution output
        )
    else:
        app = create_app(cwd=cwd, assets_dir=assets_dir, venv=venv)
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)


# For reload mode
_cwd = None
_assets_dir = None
_venv = None


def _create_app_for_reload():
    from .server import create_app

    return create_app(cwd=_cwd, assets_dir=_assets_dir, venv=_venv)


if __name__ == "__main__":
    main()
