"""
MRP HTTP Server

Implements the MRMD Runtime Protocol (MRP) over HTTP with SSE streaming.

The server exposes endpoints at /mrp/v1/* for:
- Code execution (sync and streaming)
- Completions, hover, and inspect
- Variable inspection
- Runtime reset
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
import time
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
from sse_starlette.event import ServerSentEvent

from .runtime_client import DaemonRuntimeClient
from .worker import IPythonWorker
# Runtime workers can run either:
# - locally in-process (daemon_mode=True), or
# - in an external daemon process via DaemonRuntimeClient.
#
# Runtime model is single-namespace: one server process == one REPL runtime.
from .types import (
    Capabilities,
    CapabilityFeatures,
    Environment,
    ExecuteResult,
    ExecuteError,
    InputCancelledError,
    StdinNotAllowedError,
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


class RuntimeManager:
    """Manages a single Python runtime process.

    Runtime model:
    - one server process = one REPL namespace
    - no per-request or per-document MRP sessions
    - at most one active execution at a time
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
        self.runtime: dict | None = None
        self.worker: IPythonWorker | DaemonRuntimeClient | None = None
        self._pending_inputs: dict[str, asyncio.Future] = {}
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._state_revision = 0
        self._state = "idle"
        self._current_exec_id: str | None = None

    def get_or_create_runtime(
        self,
        venv: str | None = None,
        cwd: str | None = None,
    ) -> tuple[IPythonWorker | DaemonRuntimeClient, dict]:
        """Get or create the single runtime worker."""
        effective_venv = venv or self.default_venv or _get_current_venv() or sys.prefix
        effective_cwd = cwd or self.cwd

        with self._lock:
            if self.runtime is None or self.worker is None:
                if self.daemon_mode:
                    worker = IPythonWorker(
                        cwd=effective_cwd,
                        assets_dir=self.assets_dir,
                        venv=effective_venv,
                    )
                    runtime = {
                        "id": "runtime",
                        "language": "python",
                        "created": datetime.now(timezone.utc).isoformat(),
                        "lastActivity": datetime.now(timezone.utc).isoformat(),
                        "executionCount": 0,
                        "variableCount": 0,
                        "workerType": "local",
                        "environment": {
                            "cwd": effective_cwd,
                            "virtualenv": effective_venv,
                        },
                    }
                else:
                    client = DaemonRuntimeClient(
                        runtime_id="runtime",
                        venv=effective_venv,
                        cwd=effective_cwd,
                        assets_dir=self.assets_dir,
                        auto_spawn=True,
                    )
                    daemon_info = client.get_info()
                    worker = client
                    runtime = {
                        "id": "runtime",
                        "language": "python",
                        "created": datetime.now(timezone.utc).isoformat(),
                        "lastActivity": datetime.now(timezone.utc).isoformat(),
                        "executionCount": 0,
                        "variableCount": 0,
                        "workerType": "daemon",
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

                self.worker = worker
                self.runtime = runtime

            self.runtime["lastActivity"] = datetime.now(timezone.utc).isoformat()
            return self.worker, self.runtime

    def get_runtime(self) -> dict | None:
        """Get runtime info if initialized."""
        return self.runtime

    def get_health(self) -> dict:
        runtime_name = "mrmd-python"
        execution_count = self.runtime["executionCount"] if self.runtime else 0
        return {
            "status": "ok",
            "state": self._state,
            "runtime": runtime_name,
            "currentExecId": self._current_exec_id,
            "executionCount": execution_count,
            "stateRevision": self._state_revision,
            "uptime": int(time.time() - self._started_at),
        }

    def try_begin_execution(self, exec_id: str) -> tuple[bool, str | None]:
        """Try to start an execution, returning (started, currentExecId)."""
        with self._lock:
            if self._current_exec_id is not None:
                return False, self._current_exec_id
            self._current_exec_id = exec_id
            self._state = "executing"
            if self.runtime is not None:
                self.runtime["lastActivity"] = datetime.now(timezone.utc).isoformat()
            return True, None

    def finish_execution(self) -> int:
        """Mark the active execution as complete and bump state revision."""
        with self._lock:
            self._current_exec_id = None
            self._state = "idle"
            self._state_revision += 1
            if self.runtime is not None:
                self.runtime["lastActivity"] = datetime.now(timezone.utc).isoformat()
            return self._state_revision

    def mark_waiting_input(self, waiting: bool):
        """Update runtime state while waiting for stdin."""
        with self._lock:
            if self._current_exec_id is None:
                self._state = "idle"
            else:
                self._state = "waiting_input" if waiting else "executing"
            if self.runtime is not None:
                self.runtime["lastActivity"] = datetime.now(timezone.utc).isoformat()

    def is_busy(self) -> bool:
        with self._lock:
            return self._current_exec_id is not None

    def clear_assets(self) -> bool:
        """Delete all saved assets."""
        if not self.assets_dir:
            return False
        assets_path = Path(self.assets_dir)
        if not assets_path.exists():
            return True
        for child in assets_path.iterdir():
            try:
                if child.is_file():
                    child.unlink()
            except Exception:
                pass
        return True

    def reset_runtime(self, scope: list[str] | None = None) -> dict:
        """Reset runtime state according to requested scope."""
        scope = scope or ["namespace"]
        cleared: list[str] = []

        needs_worker = any(item in scope for item in ("namespace", "history"))
        worker = None
        if needs_worker:
            worker, runtime = self.get_or_create_runtime()
        else:
            runtime = self.runtime

        if "namespace" in scope and worker is not None and hasattr(worker, "reset"):
            worker.reset()
            cleared.append("namespace")

        if "history" in scope and worker is not None and hasattr(worker, "clear_history"):
            if worker.clear_history():
                cleared.append("history")

        if "assets" in scope and self.clear_assets():
            cleared.append("assets")

        with self._lock:
            self._state_revision += 1
            if runtime is not None:
                runtime["lastActivity"] = datetime.now(timezone.utc).isoformat()
                execution_count = runtime.get("executionCount", 0)
            else:
                execution_count = 0

        return {
            "reset": True,
            "cleared": cleared,
            "stateRevision": self._state_revision,
            "executionCount": execution_count,
        }

    def shutdown(self) -> bool:
        """Shutdown runtime worker and clear metadata."""
        with self._lock:
            if self.worker is None:
                return False
            worker = self.worker
            if hasattr(worker, 'shutdown'):
                worker.shutdown()
            elif hasattr(worker, 'reset'):
                worker.reset()
            self.worker = None
            self.runtime = None
            self._state = "idle"
            self._current_exec_id = None
            return True

    def register_pending_input(self, exec_id: str, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        """Register that an execution is waiting for input."""
        future = loop.create_future()
        self._pending_inputs[exec_id] = future
        self.mark_waiting_input(True)
        return future

    def provide_input(self, exec_id: str, text: str) -> bool:
        """Provide input to a waiting execution."""
        if exec_id in self._pending_inputs:
            future = self._pending_inputs.pop(exec_id)
            if not future.done():
                future.get_loop().call_soon_threadsafe(future.set_result, text)
                self.mark_waiting_input(False)
                return True
        return False

    def cancel_pending_input(self, exec_id: str) -> bool:
        """Cancel a pending input request."""
        if exec_id in self._pending_inputs:
            future = self._pending_inputs.pop(exec_id)
            if not future.done():
                future.get_loop().call_soon_threadsafe(
                    future.set_exception,
                    InputCancelledError("Input cancelled by user")
                )
                self.mark_waiting_input(False)
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
        self.runtime_manager = RuntimeManager(
            cwd=self.cwd,
            assets_dir=self.assets_dir,
            venv=venv,
            daemon_mode=daemon_mode,
        )

    def get_capabilities(self) -> Capabilities:
        """Get server capabilities."""
        return Capabilities(
            runtime="mrmd-python",
            version="0.3.0",
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
                history=True,
                assets=True,
            ),
            environment=Environment(
                cwd=self.cwd,
                executable=sys.executable,
                virtualenv=self.venv or os.environ.get("VIRTUAL_ENV"),
            ),
        )

    # =========================================================================
    # Route Handlers
    # =========================================================================

    def _execution_in_progress_response(self, current_exec_id: str) -> JSONResponse:
        return JSONResponse(
            {
                "error": "execution_in_progress",
                "message": "An execution is already running. Use /interrupt to cancel it.",
                "execId": current_exec_id,
            },
            status_code=409,
        )

    def _runtime_busy_response(self) -> JSONResponse:
        return JSONResponse(
            {
                "error": "runtime_busy",
                "message": "Runtime is executing code. Try again shortly.",
            },
            status_code=503,
            headers={"Retry-After": "1"},
        )

    async def handle_health(self, request: Request) -> JSONResponse:
        """GET /health"""
        return JSONResponse(self.runtime_manager.get_health())

    async def handle_capabilities(self, request: Request) -> JSONResponse:
        """GET /capabilities"""
        caps = self.get_capabilities()
        return JSONResponse(_dataclass_to_dict(caps))

    async def handle_reset_runtime(self, request: Request) -> JSONResponse:
        """POST /reset"""
        if self.runtime_manager.is_busy():
            return self._execution_in_progress_response(self.runtime_manager.get_health()["currentExecId"] or "")

        body = await request.json() if request.method == "POST" else {}
        scope = body.get("scope", ["namespace"])
        if not isinstance(scope, list):
            scope = ["namespace"]
        return JSONResponse(self.runtime_manager.reset_runtime(scope))

    async def handle_execute(self, request: Request) -> JSONResponse:
        """POST /execute"""
        body = await request.json()
        code = body.get("code", "")
        store_history = body.get("storeHistory", True)
        allow_stdin = body.get("allowStdin", False)
        exec_id = body.get("execId", str(uuid.uuid4())[:8])

        if allow_stdin:
            return JSONResponse(
                {
                    "error": "invalid_request",
                    "message": "stdin is only supported with /execute/stream",
                },
                status_code=400,
            )

        started, current_exec_id = self.runtime_manager.try_begin_execution(exec_id)
        if not started:
            return self._execution_in_progress_response(current_exec_id or "")

        try:
            worker, runtime = self.runtime_manager.get_or_create_runtime()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: worker.execute(code, store_history, exec_id, False)
            )
            runtime["executionCount"] = result.executionCount
            try:
                runtime["variableCount"] = len(worker.get_variables().variables)
            except Exception:
                pass
            result.stateRevision = self.runtime_manager.finish_execution()
            return JSONResponse(_dataclass_to_dict(result))
        except Exception as e:
            revision = self.runtime_manager.finish_execution()
            result = ExecuteResult(
                success=False,
                error=ExecuteError(type=type(e).__name__, message=str(e), traceback=[]),
                stateRevision=revision,
            )
            return JSONResponse(_dataclass_to_dict(result))

    async def handle_execute_stream(self, request: Request) -> Response:
        """POST /execute/stream - SSE streaming execution"""
        body = await request.json()
        code = body.get("code", "")
        store_history = body.get("storeHistory", True)
        allow_stdin = body.get("allowStdin", True)
        exec_id = body.get("execId", str(uuid.uuid4())[:8])

        started, current_exec_id = self.runtime_manager.try_begin_execution(exec_id)
        if not started:
            return self._execution_in_progress_response(current_exec_id or "")

        try:
            worker, runtime = self.runtime_manager.get_or_create_runtime()
        except Exception as e:
            self.runtime_manager.finish_execution()
            return JSONResponse(
                {
                    "error": "internal",
                    "message": str(e),
                },
                status_code=500,
            )

        async def event_generator():
            loop = asyncio.get_running_loop()
            output_queue: asyncio.Queue = asyncio.Queue()
            result_holder = [None]
            seq = 0

            def next_seq() -> int:
                nonlocal seq
                seq += 1
                return seq

            yield {
                "event": "start",
                "data": json.dumps({
                    "execId": exec_id,
                    "seq": next_seq(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }),
            }

            def on_output(stream: str, content: str, acc: str):
                asyncio.run_coroutine_threadsafe(
                    output_queue.put({
                        "event": stream,
                        "data": {"seq": next_seq(), "content": content},
                    }),
                    loop,
                )

            def on_stdin_request(stdin_request):
                asyncio.run_coroutine_threadsafe(
                    output_queue.put({
                        "event": "stdin_request",
                        "data": {
                            "seq": next_seq(),
                            "prompt": stdin_request.prompt,
                            "password": stdin_request.password,
                            "execId": stdin_request.execId,
                        },
                    }),
                    loop,
                )
                future = self.runtime_manager.register_pending_input(exec_id, loop)

                async def wait_for_input():
                    return await future

                concurrent_future = asyncio.run_coroutine_threadsafe(wait_for_input(), loop)
                try:
                    return concurrent_future.result(timeout=300)
                except InputCancelledError:
                    raise
                except Exception as e:
                    raise RuntimeError(f"Failed to get input: {e}")
                finally:
                    self.runtime_manager.mark_waiting_input(False)

            def run_execution():
                try:
                    result = worker.execute_streaming(
                        code,
                        on_output,
                        store_history,
                        exec_id,
                        on_stdin_request=on_stdin_request if allow_stdin else None,
                        allow_stdin=allow_stdin,
                    )
                    result_holder[0] = result
                except Exception as e:
                    if hasattr(worker, "_format_exception"):
                        error = worker._format_exception(e)
                    else:
                        error = ExecuteError(type=type(e).__name__, message=str(e), traceback=[])
                    result_holder[0] = ExecuteResult(success=False, error=error)
                finally:
                    asyncio.run_coroutine_threadsafe(output_queue.put(None), loop)

            exec_thread = threading.Thread(target=run_execution, daemon=True)
            exec_thread.start()

            while True:
                item = await output_queue.get()
                if item is None:
                    break
                yield {
                    "event": item["event"],
                    "data": json.dumps(item["data"]),
                }

            exec_thread.join(timeout=5.0)

            result = result_holder[0]
            state_revision = self.runtime_manager.finish_execution()
            if result:
                runtime["executionCount"] = result.executionCount
                try:
                    runtime["variableCount"] = len(worker.get_variables().variables)
                except Exception:
                    pass
                result.stateRevision = state_revision

                for display in result.displayData:
                    yield {
                        "event": "display",
                        "data": json.dumps({
                            "seq": next_seq(),
                            "data": display.data,
                            "metadata": display.metadata,
                        }),
                    }

                for asset in result.assets:
                    yield {
                        "event": "asset",
                        "data": json.dumps({
                            "seq": next_seq(),
                            **_dataclass_to_dict(asset),
                        }),
                    }

                for warning in result.warnings:
                    yield {
                        "event": "warning",
                        "data": json.dumps({"seq": next_seq(), **warning}),
                    }

                if result.success:
                    yield {
                        "event": "result",
                        "data": json.dumps({
                            "seq": next_seq(),
                            **_dataclass_to_dict(result),
                        }),
                    }
                else:
                    error_payload = _dataclass_to_dict(result.error) if result.error else {}
                    error_payload.update({
                        "seq": next_seq(),
                        "stateRevision": state_revision,
                        "executionCount": result.executionCount,
                        "duration": result.duration,
                        "warnings": result.warnings,
                    })
                    yield {
                        "event": "error",
                        "data": json.dumps(error_payload),
                    }

            yield {"event": "done", "data": json.dumps({"seq": next_seq()})}

        return EventSourceResponse(
            event_generator(),
            ping=15,
            ping_message_factory=lambda: ServerSentEvent(comment="keepalive"),
        )

    async def handle_input(self, request: Request) -> JSONResponse:
        """POST /input - Send user input to waiting execution"""
        body = await request.json()
        exec_id = body.get("execId") or body.get("exec_id", "")
        text = body.get("text", "")

        if self.runtime_manager.provide_input(exec_id, text):
            return JSONResponse({"accepted": True})
        return JSONResponse({"accepted": False, "error": "No pending input request"})

    async def handle_input_cancel(self, request: Request) -> JSONResponse:
        """POST /input/cancel - Cancel pending input request"""
        body = await request.json()
        exec_id = body.get("execId") or body.get("exec_id", "")

        if self.runtime_manager.cancel_pending_input(exec_id):
            return JSONResponse({"cancelled": True})
        return JSONResponse({"cancelled": False, "error": "No pending input request"})

    async def handle_interrupt(self, request: Request) -> JSONResponse:
        """POST /interrupt"""
        worker, _ = self.runtime_manager.get_or_create_runtime()
        try:
            interrupted = worker.interrupt()
            return JSONResponse({"interrupted": interrupted})
        except Exception as e:
            return JSONResponse({"interrupted": False, "error": str(e)})

    async def handle_complete(self, request: Request) -> JSONResponse:
        """POST /complete"""
        if self.runtime_manager.is_busy():
            return self._runtime_busy_response()

        body = await request.json()
        code = body.get("code", "")
        cursor = body.get("cursor", len(code))
        worker, _ = self.runtime_manager.get_or_create_runtime()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: worker.complete(code, cursor))
        return JSONResponse(_dataclass_to_dict(result))

    async def handle_inspect(self, request: Request) -> JSONResponse:
        """POST /inspect"""
        if self.runtime_manager.is_busy():
            return self._runtime_busy_response()

        body = await request.json()
        code = body.get("code", "")
        cursor = body.get("cursor", len(code))
        detail = body.get("detail", 1)
        worker, _ = self.runtime_manager.get_or_create_runtime()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: worker.inspect(code, cursor, detail))
        return JSONResponse(_dataclass_to_dict(result))

    async def handle_hover(self, request: Request) -> JSONResponse:
        """POST /hover"""
        if self.runtime_manager.is_busy():
            return self._runtime_busy_response()

        body = await request.json()
        code = body.get("code", "")
        cursor = body.get("cursor", len(code))
        worker, _ = self.runtime_manager.get_or_create_runtime()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: worker.hover(code, cursor))
        return JSONResponse(_dataclass_to_dict(result))

    async def handle_variables(self, request: Request) -> JSONResponse:
        """POST /variables"""
        if self.runtime_manager.is_busy():
            return self._runtime_busy_response()

        worker, _ = self.runtime_manager.get_or_create_runtime()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, worker.get_variables)
        return JSONResponse(_dataclass_to_dict(result))

    async def handle_variable_detail(self, request: Request) -> JSONResponse:
        """POST /variables/{name}"""
        if self.runtime_manager.is_busy():
            return self._runtime_busy_response()

        name = request.path_params["name"]
        body = await request.json()
        path = body.get("path")
        worker, _ = self.runtime_manager.get_or_create_runtime()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: worker.get_variable_detail(name, path))
        return JSONResponse(_dataclass_to_dict(result))

    async def handle_is_complete(self, request: Request) -> JSONResponse:
        """POST /is_complete"""
        body = await request.json()
        code = body.get("code", "")
        worker, _ = self.runtime_manager.get_or_create_runtime()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: worker.is_complete(code))
        return JSONResponse(_dataclass_to_dict(result))

    async def handle_format(self, request: Request) -> JSONResponse:
        """POST /format"""
        body = await request.json()
        code = body.get("code", "")
        worker, _ = self.runtime_manager.get_or_create_runtime()
        loop = asyncio.get_event_loop()
        formatted, changed = await loop.run_in_executor(None, lambda: worker.format_code(code))
        return JSONResponse({"formatted": formatted, "changed": changed})

    async def handle_history(self, request: Request) -> JSONResponse:
        """POST /history"""
        body = await request.json() if request.method == "POST" else {}
        n = body.get("n", 20)
        pattern = body.get("pattern")
        before = body.get("before")
        worker, _ = self.runtime_manager.get_or_create_runtime()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: worker.get_history(n=n, pattern=pattern, before=before)
        )
        return JSONResponse(_dataclass_to_dict(result))

    async def handle_asset(self, request: Request) -> Response:
        """GET /assets/{id}"""
        asset_id = request.path_params["id"]
        if "/" in asset_id or ".." in asset_id:
            return JSONResponse({"error": "not_found", "message": "Asset not found"}, status_code=404)

        full_path = Path(self.assets_dir) / asset_id
        if not full_path.exists() or not full_path.is_file():
            return JSONResponse({"error": "not_found", "message": "Asset not found"}, status_code=404)

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

    async def handle_playground(self, request: Request) -> Response:
        """Serve the interactive API playground."""
        playground_path = Path(__file__).parent / "playground.html"
        if not playground_path.exists():
            return JSONResponse({"error": "playground not found"}, status_code=404)
        return FileResponse(playground_path, media_type="text/html")

    def create_routes(self) -> list[Route]:
        """Create all routes."""
        return [
            Route("/mrp/v1/health", self.handle_health, methods=["GET"]),
            Route("/mrp/v1/capabilities", self.handle_capabilities, methods=["GET"]),
            Route("/mrp/v1/reset", self.handle_reset_runtime, methods=["POST"]),
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
            Route("/mrp/v1/history", self.handle_history, methods=["POST"]),
            Route("/mrp/v1/assets/{id}", self.handle_asset, methods=["GET"]),
            Route("/mrp/v1/playground", self.handle_playground, methods=["GET"]),
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
                    If False, use an external daemon-backed runtime worker.
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
