"""
MRP HTTP Server

Implements the MRMD Runtime Protocol (MRP) over HTTP with SSE streaming.

The server exposes endpoints at /mrp/v1/* for:
- Code execution (sync and streaming)
- Completions, hover, and inspect
- Variable inspection
- Session management
- Asset serving (for matplotlib figures, HTML output, etc.)

Usage:
    from mrmd_python import create_app
    app = create_app(cwd="/path/to/project")

Or via CLI:
    mrmd-python --port 8000
"""

import json
import sys
import os
import asyncio
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, FileResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .runtime_client import DaemonRuntimeClient
from .runtime_daemon import list_runtimes, is_runtime_alive, kill_runtime
from .worker import IPythonWorker
# All sessions use independent daemon runtimes for:
# - GPU memory isolation (critical for vLLM)
# - Process survival (daemon persists if orchestrator dies)
# - Registry in ~/.mrmd/runtimes/ for discovery
#
# Exception: When running IN a daemon (daemon_mode=True), we use the local
# IPythonWorker to avoid recursive daemon spawning.
from .types import (
    Capabilities,
    CapabilityFeatures,
    Environment,
    ExecuteResult,
    InputCancelledError,
)


def _get_current_venv() -> str | None:
    """Get the current Python's virtual environment path.

    Returns sys.prefix if we're in a venv, otherwise None.
    """
    # Check if we're in a virtual environment
    # sys.prefix != sys.base_prefix when in a venv
    if sys.prefix != sys.base_prefix:
        return sys.prefix
    # Also check VIRTUAL_ENV env var
    return os.environ.get("VIRTUAL_ENV")


class SessionManager:
    """Manages multiple Python sessions.

    Two modes of operation:
    1. daemon_mode=True: Used when running INSIDE a daemon process.
       Uses IPythonWorker directly for local execution.
       This is a single, independent Python runtime.

    2. daemon_mode=False (default): Used by orchestrators.
       Each session spawns an independent daemon process via DaemonRuntimeClient.
       Daemons survive if orchestrator dies, variables persist, GPU memory isolated.
    """

    def __init__(
        self,
        cwd: str | None = None,
        assets_dir: str | None = None,
        venv: str | None = None,
        daemon_mode: bool = False,
    ):
        self.cwd = cwd
        self.assets_dir = assets_dir
        self.default_venv = venv
        self.daemon_mode = daemon_mode
        self.sessions: dict[str, dict] = {}
        # In daemon_mode: workers dict holds IPythonWorker instances
        # In orchestrator mode: workers dict holds DaemonRuntimeClient instances
        self.workers: dict[str, IPythonWorker | DaemonRuntimeClient] = {}
        self._pending_inputs: dict[str, asyncio.Future] = {}
        self._lock = threading.Lock()

    def get_or_create_session(
        self,
        session_id: str,
        venv: str | None = None,
        cwd: str | None = None,
    ) -> tuple[IPythonWorker | DaemonRuntimeClient, dict]:
        """
        Get or create a session.

        Behavior depends on daemon_mode:
        - daemon_mode=True: Creates local IPythonWorker (for use inside daemon)
        - daemon_mode=False: Creates DaemonRuntimeClient (spawns independent daemon)

        Args:
            session_id: Unique session identifier
            venv: Path to virtual environment. If not provided, auto-detects.
            cwd: Working directory override for this session.

        Returns:
            Tuple of (worker, session_info)
        """
        print(f"[MRPServer.get_or_create_session] session_id={session_id}", flush=True)
        print(f"[MRPServer.get_or_create_session] venv param={venv}", flush=True)
        print(f"[MRPServer.get_or_create_session] self.default_venv={self.default_venv}", flush=True)
        print(f"[MRPServer.get_or_create_session] _get_current_venv()={_get_current_venv()}", flush=True)
        print(f"[MRPServer.get_or_create_session] sys.prefix={sys.prefix}", flush=True)

        effective_venv = venv or self.default_venv or _get_current_venv() or sys.prefix
        effective_cwd = cwd or self.cwd

        print(f"[MRPServer.get_or_create_session] effective_venv={effective_venv}", flush=True)
        print(f"[MRPServer.get_or_create_session] daemon_mode={self.daemon_mode}", flush=True)

        with self._lock:
            if session_id not in self.sessions:
                print(f"[MRPServer.get_or_create_session] Creating NEW session", flush=True)
                if self.daemon_mode:
                    # DAEMON MODE: Use local IPythonWorker
                    # This is the actual execution engine inside the daemon
                    print(f"[MRPServer.get_or_create_session] Creating IPythonWorker with venv={effective_venv}", flush=True)
                    worker = IPythonWorker(
                        cwd=effective_cwd,
                        assets_dir=self.assets_dir,
                        venv=effective_venv,
                    )
                    self.sessions[session_id] = {
                        "id": session_id,
                        "language": "python",
                        "created": datetime.now(timezone.utc).isoformat(),
                        "lastActivity": datetime.now(timezone.utc).isoformat(),
                        "executionCount": 0,
                        "variableCount": 0,
                        "workerType": "local",  # Local IPython worker
                        "environment": {
                            "cwd": effective_cwd,
                            "virtualenv": effective_venv,
                        },
                    }
                else:
                    # ORCHESTRATOR MODE: Spawn independent daemon via HTTP client
                    client = DaemonRuntimeClient(
                        runtime_id=session_id,
                        venv=effective_venv,
                        cwd=effective_cwd,
                        assets_dir=self.assets_dir,
                        auto_spawn=True,
                    )
                    daemon_info = client.get_info()
                    worker = client
                    self.sessions[session_id] = {
                        "id": session_id,
                        "language": "python",
                        "created": datetime.now(timezone.utc).isoformat(),
                        "lastActivity": datetime.now(timezone.utc).isoformat(),
                        "executionCount": 0,
                        "variableCount": 0,
                        "workerType": "daemon",  # Independent daemon process
                        "daemon": {
                            "pid": daemon_info.get("pid"),
                            "port": daemon_info.get("port"),
                            "url": daemon_info.get("url"),
                        },
                        "environment": {
                            "cwd": effective_cwd,
                            "virtualenv": effective_venv,
                        },
                    }

                self.workers[session_id] = worker
            else:
                print(f"[MRPServer.get_or_create_session] Using EXISTING session", flush=True)
                existing_worker = self.workers.get(session_id)
                if existing_worker and hasattr(existing_worker, 'venv'):
                    print(f"[MRPServer.get_or_create_session] Existing worker venv={existing_worker.venv}", flush=True)

            session = self.sessions[session_id]
            session["lastActivity"] = datetime.now(timezone.utc).isoformat()
            return self.workers[session_id], session

    def get_session(self, session_id: str) -> dict | None:
        """Get session info."""
        return self.sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        """List all sessions."""
        return list(self.sessions.values())

    def destroy_session(self, session_id: str) -> bool:
        """
        Destroy a session and release all resources.

        In daemon mode: Resets the local IPythonWorker
        In orchestrator mode: Kills the daemon process, releasing GPU/VRAM
        """
        with self._lock:
            if session_id in self.sessions:
                if session_id in self.workers:
                    worker = self.workers[session_id]
                    if hasattr(worker, 'shutdown'):
                        worker.shutdown()  # DaemonRuntimeClient - kills daemon
                    elif hasattr(worker, 'reset'):
                        worker.reset()  # IPythonWorker - just reset
                    del self.workers[session_id]

                del self.sessions[session_id]
                return True
            return False

    def get_worker_info(self, session_id: str) -> dict | None:
        """Get info about the worker for a session."""
        if session_id not in self.workers:
            return None
        return self.workers[session_id].get_info()

    def register_pending_input(self, exec_id: str, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        """Register that an execution is waiting for input."""
        future = loop.create_future()
        self._pending_inputs[exec_id] = future
        return future

    def provide_input(self, exec_id: str, text: str) -> bool:
        """Provide input to a waiting execution."""
        if exec_id in self._pending_inputs:
            future = self._pending_inputs.pop(exec_id)
            if not future.done():
                future.get_loop().call_soon_threadsafe(future.set_result, text)
                return True
        return False

    def cancel_pending_input(self, exec_id: str) -> bool:
        """Cancel a pending input request.

        This is called when the user dismisses the input field (e.g., cancels
        execution, navigates away) to unblock the waiting worker thread.
        """
        if exec_id in self._pending_inputs:
            future = self._pending_inputs.pop(exec_id)
            if not future.done():
                # Set exception to unblock the waiting worker
                future.get_loop().call_soon_threadsafe(
                    future.set_exception,
                    InputCancelledError("Input cancelled by user")
                )
                return True
        return False


class MRPServer:
    """MRP HTTP Server."""

    def __init__(
        self,
        cwd: str | None = None,
        assets_dir: str | None = None,
        venv: str | None = None,
        daemon_mode: bool = False,
    ):
        self.cwd = cwd or os.getcwd()
        self.assets_dir = assets_dir or os.path.join(self.cwd, ".mrmd-assets")
        self.venv = venv
        self.daemon_mode = daemon_mode
        self.session_manager = SessionManager(
            cwd=self.cwd,
            assets_dir=self.assets_dir,
            venv=venv,
            daemon_mode=daemon_mode,
        )

    def get_capabilities(self) -> Capabilities:
        """Get server capabilities."""
        return Capabilities(
            runtime="mrmd-python",
            version="0.1.0",
            languages=["python", "py", "python3"],
            features=CapabilityFeatures(
                execute=True,
                executeStream=True,
                interrupt=True,
                complete=True,
                inspect=True,
                hover=True,
                variables=True,
                variableExpand=True,
                reset=True,
                isComplete=True,
                format=True,
                assets=True,
            ),
            defaultSession="default",
            maxSessions=10,
            environment=Environment(
                cwd=self.cwd,
                executable=sys.executable,
                virtualenv=self.venv or os.environ.get("VIRTUAL_ENV"),
            ),
        )

    # =========================================================================
    # Route Handlers
    # =========================================================================

    async def handle_capabilities(self, request: Request) -> JSONResponse:
        """GET /capabilities"""
        caps = self.get_capabilities()
        return JSONResponse(_dataclass_to_dict(caps))

    async def handle_list_sessions(self, request: Request) -> JSONResponse:
        """GET /sessions"""
        sessions = self.session_manager.list_sessions()
        return JSONResponse({"sessions": sessions})

    async def handle_create_session(self, request: Request) -> JSONResponse:
        """POST /sessions

        Create a new session with optional environment configuration.

        Request body:
            id: Optional session ID (generated if not provided)
            language: Language (default: python)
            environment:
                cwd: Working directory
                virtualenv: Path to venv (enables subprocess isolation)
                executable: Python executable (ignored, derived from venv)
                env: Environment variables (not yet implemented)
            dependencies: Package dependencies (not yet implemented)
        """
        body = await request.json()
        session_id = body.get("id", str(uuid.uuid4())[:8])

        # Extract environment config
        env_config = body.get("environment", {})
        venv = env_config.get("virtualenv")
        cwd = env_config.get("cwd")

        worker, session = self.session_manager.get_or_create_session(
            session_id,
            venv=venv,
            cwd=cwd,
        )
        return JSONResponse(session)

    async def handle_get_session(self, request: Request) -> JSONResponse:
        """GET /sessions/{id}"""
        session_id = request.path_params["id"]
        session = self.session_manager.get_session(session_id)
        if not session:
            return JSONResponse({"error": "Session not found"}, status_code=404)
        return JSONResponse(session)

    async def handle_delete_session(self, request: Request) -> JSONResponse:
        """DELETE /sessions/{id}"""
        session_id = request.path_params["id"]
        if self.session_manager.destroy_session(session_id):
            return JSONResponse({"success": True})
        return JSONResponse({"error": "Session not found"}, status_code=404)

    async def handle_reset_session(self, request: Request) -> JSONResponse:
        """POST /sessions/{id}/reset"""
        session_id = request.path_params["id"]
        worker, session = self.session_manager.get_or_create_session(session_id)
        worker.reset()
        session["executionCount"] = 0
        session["variableCount"] = 0
        return JSONResponse({"success": True})

    async def handle_execute(self, request: Request) -> JSONResponse:
        """POST /execute"""
        body = await request.json()
        code = body.get("code", "")
        session_id = body.get("session", "default")
        store_history = body.get("storeHistory", True)
        exec_id = body.get("execId", str(uuid.uuid4())[:8])

        worker, session = self.session_manager.get_or_create_session(session_id)

        # Run in thread pool to not block
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: worker.execute(code, store_history, exec_id)
        )

        session["executionCount"] = result.executionCount
        session["variableCount"] = len(worker.get_variables().variables)

        return JSONResponse(_dataclass_to_dict(result))

    async def handle_execute_stream(self, request: Request) -> EventSourceResponse:
        """POST /execute/stream - SSE streaming execution"""
        print(f"[DEBUG] handle_execute_stream called", flush=True)
        body = await request.json()
        code = body.get("code", "")
        session_id = body.get("session", "default")
        store_history = body.get("storeHistory", True)
        exec_id = body.get("execId", str(uuid.uuid4())[:8])
        print(f"[DEBUG] session_id={session_id}, code length={len(code)}, exec_id={exec_id}", flush=True)

        worker, session = self.session_manager.get_or_create_session(session_id)
        print(f"[DEBUG] Got worker, starting event generator", flush=True)

        async def event_generator():
            # Capture the event loop for use in background threads
            loop = asyncio.get_running_loop()

            # Send start event
            yield {
                "event": "start",
                "data": json.dumps({
                    "execId": exec_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }),
            }

            # Queue for output from the worker thread
            output_queue: asyncio.Queue = asyncio.Queue()
            result_holder = [None]
            accumulated = {"stdout": "", "stderr": ""}

            def on_output(stream: str, content: str, acc: str):
                """Called from worker thread for each output chunk."""
                accumulated[stream] = acc
                asyncio.run_coroutine_threadsafe(
                    output_queue.put({
                        "event": stream,
                        "data": {"content": content, "accumulated": acc},
                    }),
                    loop,  # Use captured loop
                )

            def on_stdin_request(request):
                """Called from worker thread when input() is called.

                This function blocks until input is provided via POST /input.
                """
                from .types import StdinRequest

                # Send stdin_request event to client
                asyncio.run_coroutine_threadsafe(
                    output_queue.put({
                        "event": "stdin_request",
                        "data": {
                            "prompt": request.prompt,
                            "password": request.password,
                            "execId": request.execId,
                        },
                    }),
                    loop,
                )

                # Register that we're waiting for input and get a future
                future = self.session_manager.register_pending_input(exec_id, loop)

                # Wait for the input (blocking - we're in a worker thread)
                # Use run_coroutine_threadsafe to wait on the future from this thread
                async def wait_for_input():
                    return await future

                concurrent_future = asyncio.run_coroutine_threadsafe(wait_for_input(), loop)

                try:
                    # Wait up to 5 minutes for input
                    response = concurrent_future.result(timeout=300)
                    return response
                except InputCancelledError:
                    # Re-raise InputCancelledError so the worker can handle it
                    raise
                except Exception as e:
                    raise RuntimeError(f"Failed to get input: {e}")

            def run_execution():
                """Run execution in thread."""
                try:
                    result = worker.execute_streaming(
                        code, on_output, store_history, exec_id,
                        on_stdin_request=on_stdin_request
                    )
                    result_holder[0] = result
                except Exception as e:
                    result_holder[0] = ExecuteResult(
                        success=False,
                        error=worker._format_exception(e),
                    )
                finally:
                    asyncio.run_coroutine_threadsafe(
                        output_queue.put(None),  # Signal completion
                        loop,  # Use captured loop
                    )

            # Start execution in background thread
            exec_thread = threading.Thread(target=run_execution, daemon=True)
            exec_thread.start()

            # Stream output events
            while True:
                try:
                    item = await asyncio.wait_for(output_queue.get(), timeout=60.0)
                    if item is None:
                        break
                    yield {
                        "event": item["event"],
                        "data": json.dumps(item["data"]),
                    }
                except asyncio.TimeoutError:
                    # Keep connection alive
                    yield {"event": "ping", "data": "{}"}

            # Wait for thread to finish
            exec_thread.join(timeout=5.0)

            result = result_holder[0]
            if result:
                session["executionCount"] = result.executionCount
                session["variableCount"] = len(worker.get_variables().variables)

                if result.success:
                    yield {
                        "event": "result",
                        "data": json.dumps(_dataclass_to_dict(result)),
                    }
                else:
                    yield {
                        "event": "error",
                        "data": json.dumps(_dataclass_to_dict(result.error) if result.error else {}),
                    }

            yield {"event": "done", "data": "{}"}

        return EventSourceResponse(event_generator())

    async def handle_input(self, request: Request) -> JSONResponse:
        """POST /input - Send user input to waiting execution"""
        body = await request.json()
        exec_id = body.get("exec_id", "")
        text = body.get("text", "")

        if self.session_manager.provide_input(exec_id, text):
            return JSONResponse({"accepted": True})
        return JSONResponse({"accepted": False, "error": "No pending input request"})

    async def handle_input_cancel(self, request: Request) -> JSONResponse:
        """POST /input/cancel - Cancel pending input request

        Called when the user dismisses the input field without providing input.
        This unblocks the waiting execution and marks it as cancelled.
        """
        body = await request.json()
        exec_id = body.get("exec_id", "")

        if self.session_manager.cancel_pending_input(exec_id):
            return JSONResponse({"cancelled": True})
        return JSONResponse({"cancelled": False, "error": "No pending input request"})

    async def handle_interrupt(self, request: Request) -> JSONResponse:
        """POST /interrupt

        Interrupt currently running code in a session.
        Sends SIGINT to subprocess workers or sets interrupt flag for local workers.
        """
        body = await request.json()
        session_id = body.get("session", "default")

        # Get the worker for this session (don't create if doesn't exist)
        session = self.session_manager.get_session(session_id)
        if not session:
            return JSONResponse({"interrupted": False, "error": "Session not found"})

        worker = self.session_manager.workers.get(session_id)
        if not worker:
            return JSONResponse({"interrupted": False, "error": "Worker not found"})

        # Call interrupt on the worker
        try:
            interrupted = worker.interrupt()
            return JSONResponse({"interrupted": interrupted})
        except Exception as e:
            return JSONResponse({"interrupted": False, "error": str(e)})

    async def handle_complete(self, request: Request) -> JSONResponse:
        """POST /complete"""
        body = await request.json()
        code = body.get("code", "")
        cursor = body.get("cursor", len(code))
        session_id = body.get("session", "default")

        worker, _ = self.session_manager.get_or_create_session(session_id)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: worker.complete(code, cursor)
        )

        return JSONResponse(_dataclass_to_dict(result))

    async def handle_inspect(self, request: Request) -> JSONResponse:
        """POST /inspect"""
        body = await request.json()
        code = body.get("code", "")
        cursor = body.get("cursor", len(code))
        session_id = body.get("session", "default")
        detail = body.get("detail", 1)

        worker, _ = self.session_manager.get_or_create_session(session_id)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: worker.inspect(code, cursor, detail)
        )

        return JSONResponse(_dataclass_to_dict(result))

    async def handle_hover(self, request: Request) -> JSONResponse:
        """POST /hover"""
        body = await request.json()
        code = body.get("code", "")
        cursor = body.get("cursor", len(code))
        session_id = body.get("session", "default")

        worker, _ = self.session_manager.get_or_create_session(session_id)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: worker.hover(code, cursor)
        )

        return JSONResponse(_dataclass_to_dict(result))

    async def handle_variables(self, request: Request) -> JSONResponse:
        """POST /variables"""
        body = await request.json()
        session_id = body.get("session", "default")

        worker, _ = self.session_manager.get_or_create_session(session_id)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, worker.get_variables)

        return JSONResponse(_dataclass_to_dict(result))

    async def handle_variable_detail(self, request: Request) -> JSONResponse:
        """POST /variables/{name}"""
        name = request.path_params["name"]
        body = await request.json()
        session_id = body.get("session", "default")
        path = body.get("path")

        worker, _ = self.session_manager.get_or_create_session(session_id)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: worker.get_variable_detail(name, path)
        )

        return JSONResponse(_dataclass_to_dict(result))

    async def handle_is_complete(self, request: Request) -> JSONResponse:
        """POST /is_complete"""
        body = await request.json()
        code = body.get("code", "")
        session_id = body.get("session", "default")

        worker, _ = self.session_manager.get_or_create_session(session_id)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: worker.is_complete(code)
        )

        return JSONResponse(_dataclass_to_dict(result))

    async def handle_format(self, request: Request) -> JSONResponse:
        """POST /format"""
        body = await request.json()
        code = body.get("code", "")
        session_id = body.get("session", "default")

        worker, _ = self.session_manager.get_or_create_session(session_id)

        loop = asyncio.get_event_loop()
        formatted, changed = await loop.run_in_executor(
            None, lambda: worker.format_code(code)
        )

        return JSONResponse({"formatted": formatted, "changed": changed})

    async def handle_asset(self, request: Request) -> Response:
        """GET /assets/{path}"""
        asset_path = request.path_params["path"]
        full_path = Path(self.assets_dir) / asset_path

        if not full_path.exists():
            return JSONResponse({"error": "Asset not found"}, status_code=404)

        # Determine content type
        suffix = full_path.suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".html": "text/html",
            ".json": "application/json",
        }
        content_type = content_types.get(suffix, "application/octet-stream")

        return FileResponse(full_path, media_type=content_type)

    def create_routes(self) -> list[Route]:
        """Create all routes."""
        return [
            Route("/mrp/v1/capabilities", self.handle_capabilities, methods=["GET"]),
            Route("/mrp/v1/sessions", self.handle_list_sessions, methods=["GET"]),
            Route("/mrp/v1/sessions", self.handle_create_session, methods=["POST"]),
            Route("/mrp/v1/sessions/{id}", self.handle_get_session, methods=["GET"]),
            Route("/mrp/v1/sessions/{id}", self.handle_delete_session, methods=["DELETE"]),
            Route("/mrp/v1/sessions/{id}/reset", self.handle_reset_session, methods=["POST"]),
            Route("/mrp/v1/execute", self.handle_execute, methods=["POST"]),
            Route("/mrp/v1/execute/stream", self.handle_execute_stream, methods=["POST"]),
            Route("/mrp/v1/input", self.handle_input, methods=["POST"]),
            Route("/mrp/v1/input/cancel", self.handle_input_cancel, methods=["POST"]),
            Route("/mrp/v1/interrupt", self.handle_interrupt, methods=["POST"]),
            Route("/mrp/v1/complete", self.handle_complete, methods=["POST"]),
            Route("/mrp/v1/inspect", self.handle_inspect, methods=["POST"]),
            Route("/mrp/v1/hover", self.handle_hover, methods=["POST"]),
            Route("/mrp/v1/variables", self.handle_variables, methods=["POST"]),
            Route("/mrp/v1/variables/{name}", self.handle_variable_detail, methods=["POST"]),
            Route("/mrp/v1/is_complete", self.handle_is_complete, methods=["POST"]),
            Route("/mrp/v1/format", self.handle_format, methods=["POST"]),
            Route("/mrp/v1/assets/{path:path}", self.handle_asset, methods=["GET"]),
        ]


def _dataclass_to_dict(obj: Any) -> dict:
    """Convert dataclass to dict, handling nested dataclasses."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)
            result[field_name] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj


def create_app(
    cwd: str | None = None,
    assets_dir: str | None = None,
    venv: str | None = None,
    daemon_mode: bool = False,
) -> Starlette:
    """Create the MRP server application.

    Args:
        cwd: Working directory for code execution
        assets_dir: Directory for saving assets (plots, etc.)
        venv: Path to virtual environment to use for code execution.
        daemon_mode: If True, use local IPythonWorker (for daemon process).
                    If False, spawn independent daemon processes for each session.
    """
    server = MRPServer(cwd=cwd, assets_dir=assets_dir, venv=venv, daemon_mode=daemon_mode)

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    app = Starlette(
        routes=server.create_routes(),
        middleware=middleware,
    )

    return app
