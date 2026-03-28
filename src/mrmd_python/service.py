"""
RuntimeService — the bridge between IPythonWorker and MCP tools.

Owns:
- execution lifecycle (state machine, event log, input handling)
- health / capabilities / state revision
- delegation to IPythonWorker for actual computation

This is the single shared service that all MCP tools call.
"""

import asyncio
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .types import (
    Asset,
    Capabilities,
    CapabilityFeatures,
    CompleteResult,
    DisplayData,
    Environment,
    ExecuteError,
    ExecuteResult,
    ExecutionEvent,
    ExecutionEventsResponse,
    ExecutionResult,
    ExecutionState,
    FormatResult,
    Health,
    HoverResult,
    InputCancelledError,
    InputRequestInfo,
    InspectResult,
    IsCompleteResult,
    StdinNotAllowedError,
    StdinRequest,
    Variable,
    VariableDetail,
    VariablesResult,
    Warning,
    HistoryResult,
)
from .worker import IPythonWorker


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass to dict."""
    if hasattr(obj, "__dataclass_fields__"):
        result = {}
        for name in obj.__dataclass_fields__:
            value = getattr(obj, name)
            result[name] = _dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


class Execution:
    """A single execution with state machine and event log."""

    def __init__(self, execution_id: str, client_execution_id: str | None = None):
        self.execution_id = execution_id
        self.client_execution_id = client_execution_id
        self.status: str = "working"
        self.status_message: str | None = "Execution started"
        self.created_at: str = _now()
        self.last_updated_at: str = self.created_at
        self.events: list[ExecutionEvent] = []
        self.seq: int = 0
        self.input_request: InputRequestInfo | None = None
        self.warnings: list[Warning] = []
        self.engine_result: ExecuteResult | None = None

        # Input handling
        self._input_future: asyncio.Future | None = None
        self._input_loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def append_event(self, event_type: str, **kwargs) -> ExecutionEvent:
        """Append an event to the log."""
        event = ExecutionEvent(
            seq=self._next_seq(),
            timestamp=_now(),
            type=event_type,
            **kwargs,
        )
        self.events.append(event)
        self.last_updated_at = event.timestamp
        return event

    def set_status(self, status: str, message: str | None = None):
        self.status = status
        self.status_message = message
        self.last_updated_at = _now()

    def to_state(self, state_revision: int) -> ExecutionState:
        return ExecutionState(
            executionId=self.execution_id,
            clientExecutionId=self.client_execution_id,
            status=self.status,
            statusMessage=self.status_message,
            createdAt=self.created_at,
            lastUpdatedAt=self.last_updated_at,
            pollIntervalMs=500 if not self.is_terminal else None,
            ttlMs=3600000,
            currentSeq=self.seq,
            stateRevision=state_revision,
            inputRequest=self.input_request,
            warnings=self.warnings,
        )

    def to_result(self, state_revision: int) -> ExecutionResult:
        """Convert engine result to protocol result."""
        er = self.engine_result
        if er is None:
            return ExecutionResult(
                executionId=self.execution_id,
                status=self.status,
                success=False,
                stateRevision=state_revision,
            )
        # Convert engine warnings to protocol warnings
        warnings = [
            Warning(code=w.get("code", "unknown"), message=w.get("message", ""))
            for w in er.warnings
        ] if er.warnings else []

        status = self.status
        if status not in ("completed", "failed", "cancelled"):
            status = "completed" if er.success else "failed"

        return ExecutionResult(
            executionId=self.execution_id,
            status=status,
            success=er.success,
            stdout=er.stdout,
            stderr=er.stderr,
            result=er.result,
            error=er.error,
            displayData=er.displayData,
            assets=er.assets,
            executionCount=er.executionCount,
            stateRevision=state_revision,
            durationMs=er.duration,
            imports=er.imports,
            warnings=warnings,
        )


class RuntimeService:
    """
    The shared runtime service. All MCP tools call this.

    Owns one IPythonWorker and manages execution lifecycle.
    """

    def __init__(
        self,
        cwd: str | None = None,
        assets_dir: str | None = None,
        venv: str | None = None,
    ):
        self.cwd = cwd or os.getcwd()
        self.assets_dir = assets_dir or os.path.join(self.cwd, ".mrmd-assets")
        self.venv = venv

        # Worker
        self._worker: IPythonWorker | None = None
        self._lock = threading.Lock()

        # State
        self._started_at = time.time()
        self._state_revision = 0
        self._execution_count = 0

        # Active execution
        self._current: Execution | None = None
        self._executions: dict[str, Execution] = {}

    def _get_worker(self) -> IPythonWorker:
        with self._lock:
            if self._worker is None:
                venv = self.venv or self._detect_venv()
                self._worker = IPythonWorker(
                    cwd=self.cwd,
                    assets_dir=self.assets_dir,
                    venv=venv,
                )
            return self._worker

    def _detect_venv(self) -> str | None:
        if sys.prefix != sys.base_prefix:
            return sys.prefix
        return os.environ.get("VIRTUAL_ENV")

    def _bump_revision(self) -> int:
        self._state_revision += 1
        return self._state_revision

    # ── Health & Capabilities ────────────────────────────────────

    def get_health(self) -> dict:
        current_id = self._current.execution_id if self._current and not self._current.is_terminal else None
        if current_id:
            state = "waiting_input" if self._current.status == "input_required" else "executing"
        else:
            state = "idle"
        return _dataclass_to_dict(Health(
            state=state,
            currentExecutionId=current_id,
            executionCount=self._execution_count,
            stateRevision=self._state_revision,
            uptimeSeconds=int(time.time() - self._started_at),
        ))

    def get_capabilities(self) -> dict:
        venv = self.venv or self._detect_venv()
        worker = self._get_worker()
        has_format = True
        try:
            import black  # noqa: F401
        except ImportError:
            has_format = False

        return _dataclass_to_dict(Capabilities(
            features=CapabilityFeatures(format=has_format),
            environment=Environment(
                cwd=self.cwd,
                executable=sys.executable,
                virtualenv=venv,
            ),
        ))

    # ── Reset ────────────────────────────────────────────────────

    def reset(self, scope: list[str] | None = None) -> dict:
        if self._current and not self._current.is_terminal:
            raise RuntimeError("Cannot reset while execution is active")

        scope = scope or ["namespace"]
        worker = self._get_worker()
        cleared = []

        if "namespace" in scope and hasattr(worker, "reset"):
            worker.reset()
            cleared.append("namespace")

        if "history" in scope and hasattr(worker, "clear_history"):
            if worker.clear_history():
                cleared.append("history")

        if "assets" in scope:
            self._clear_assets()
            cleared.append("assets")

        rev = self._bump_revision()
        return {
            "reset": True,
            "cleared": cleared,
            "stateRevision": rev,
            "executionCount": self._execution_count,
        }

    def _clear_assets(self):
        p = Path(self.assets_dir)
        if p.exists():
            for child in p.iterdir():
                if child.is_file():
                    try:
                        child.unlink()
                    except Exception:
                        pass

    # ── Execution ────────────────────────────────────────────────

    def start_execution(
        self,
        code: str,
        store_history: bool = True,
        silent: bool = False,
        allow_input: bool = True,
        client_execution_id: str | None = None,
        **kwargs,
    ) -> dict:
        """Start a detached execution. Returns execution state."""
        if self._current and not self._current.is_terminal:
            raise RuntimeError(f"Execution {self._current.execution_id} is already active")

        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        execution = Execution(execution_id, client_execution_id)
        self._current = execution
        self._executions[execution_id] = execution

        # Launch in background thread
        thread = threading.Thread(
            target=self._run_execution,
            args=(execution, code, store_history, silent, allow_input),
            daemon=True,
        )
        thread.start()

        return {
            "accepted": True,
            "execution": _dataclass_to_dict(execution.to_state(self._state_revision)),
        }

    def execute_sync(
        self,
        code: str,
        store_history: bool = True,
        silent: bool = False,
        allow_input: bool = True,
        client_execution_id: str | None = None,
        **kwargs,
    ) -> dict:
        """Execute and wait for completion. Returns final execution result."""
        accepted = self.start_execution(
            code=code,
            store_history=store_history,
            silent=silent,
            allow_input=allow_input,
            client_execution_id=client_execution_id,
            **kwargs,
        )
        execution_id = accepted["execution"]["executionId"]
        execution = self._executions[execution_id]

        # Wait for terminal state
        while not execution.is_terminal:
            time.sleep(0.05)

        return self.get_execution_result(execution_id)

    def execute_until_input_or_done(
        self,
        code: str,
        store_history: bool = True,
        silent: bool = False,
        allow_input: bool = True,
        client_execution_id: str | None = None,
        **kwargs,
    ) -> tuple[str, dict]:
        """Execute and wait for terminal OR input_required.

        Returns (status, data) where:
        - ("completed"|"failed"|"cancelled", execution_result_dict)
        - ("input_required", execution_state_dict)
        """
        accepted = self.start_execution(
            code=code,
            store_history=store_history,
            silent=silent,
            allow_input=allow_input,
            client_execution_id=client_execution_id,
            **kwargs,
        )
        execution_id = accepted["execution"]["executionId"]
        execution = self._executions[execution_id]

        # Wait for terminal or input_required
        while not execution.is_terminal and execution.status != "input_required":
            time.sleep(0.05)

        if execution.is_terminal:
            return execution.status, self.get_execution_result(execution_id)
        else:
            return "input_required", self.get_execution(execution_id)

    def provide_input_and_wait(
        self,
        text: str,
    ) -> tuple[str, dict]:
        """Provide input to current execution and wait for terminal or next input.

        Returns (status, data) same as execute_until_input_or_done.
        """
        execution = self._current
        if not execution or execution.is_terminal:
            raise RuntimeError("No active execution")
        if execution.status != "input_required" or execution.input_request is None:
            raise RuntimeError("No input request pending")

        # Feed input
        input_request_id = execution.input_request.inputRequestId
        self.provide_input(execution.execution_id, input_request_id, text)

        # Wait for terminal or next input_required
        while not execution.is_terminal and execution.status != "input_required":
            time.sleep(0.05)

        if execution.is_terminal:
            return execution.status, self.get_execution_result(execution.execution_id)
        else:
            return "input_required", self.get_execution(execution.execution_id)

    def _run_execution(
        self,
        execution: Execution,
        code: str,
        store_history: bool,
        silent: bool,
        allow_input: bool,
    ):
        """Run in background thread. Populates execution events and result."""
        worker = self._get_worker()

        def on_output(stream: str, content: str, accumulated: str):
            execution.append_event(stream, content=content)

        def on_stdin_request(stdin_req: StdinRequest):
            req_id = f"input_{uuid.uuid4().hex[:8]}"
            info = InputRequestInfo(
                inputRequestId=req_id,
                prompt=stdin_req.prompt,
                password=stdin_req.password,
            )
            execution.input_request = info
            execution.set_status("input_required", f"Waiting for input: {stdin_req.prompt}")
            execution.append_event("input_requested", inputRequest=info)

            # Wait for input via asyncio future
            loop = asyncio.new_event_loop()
            future = loop.create_future()
            execution._input_future = future
            execution._input_loop = loop

            try:
                loop.run_until_complete(future)
                result = future.result()
                execution.input_request = None
                execution.set_status("working", "Execution resumed")
                return result
            except InputCancelledError:
                execution.input_request = None
                raise
            finally:
                execution._input_future = None
                execution._input_loop = None
                loop.close()

        try:
            result = worker.execute_streaming(
                code,
                on_output,
                store_history,
                execution.execution_id,
                on_stdin_request=on_stdin_request if allow_input else None,
                allow_stdin=allow_input,
            )
            execution.engine_result = result
            self._execution_count += 1
            rev = self._bump_revision()
            execution.set_status(
                "completed" if result.success else "failed",
                None if result.success else (result.error.message if result.error else "Execution failed"),
            )

            # Append display/asset events from result
            for dd in result.displayData:
                execution.append_event("display", displayData=dd)
            for asset in result.assets:
                execution.append_event("asset", asset=asset)
            for w in result.warnings:
                execution.append_event("warning", warning=Warning(
                    code=w.get("code", "unknown"),
                    message=w.get("message", ""),
                ))

        except Exception as e:
            rev = self._bump_revision()
            error = ExecuteError(type=type(e).__name__, message=str(e))
            execution.engine_result = ExecuteResult(
                success=False,
                error=error,
                stateRevision=rev,
            )
            execution.set_status("failed", str(e))

    def get_execution(self, execution_id: str) -> dict:
        execution = self._executions.get(execution_id)
        if not execution:
            raise KeyError(f"Execution {execution_id} not found")
        return _dataclass_to_dict(execution.to_state(self._state_revision))

    def get_execution_events(
        self,
        execution_id: str,
        after_seq: int = 0,
        limit: int = 100,
    ) -> dict:
        execution = self._executions.get(execution_id)
        if not execution:
            raise KeyError(f"Execution {execution_id} not found")

        events = [e for e in execution.events if e.seq > after_seq][:limit]
        next_seq = events[-1].seq if events else after_seq
        return _dataclass_to_dict(ExecutionEventsResponse(
            executionId=execution_id,
            events=events,
            nextSeq=next_seq + 1,
            done=execution.is_terminal and not [e for e in execution.events if e.seq > next_seq],
        ))

    def get_execution_result(self, execution_id: str) -> dict:
        execution = self._executions.get(execution_id)
        if not execution:
            raise KeyError(f"Execution {execution_id} not found")
        if not execution.is_terminal:
            raise RuntimeError(f"Execution {execution_id} is not terminal (status: {execution.status})")
        return _dataclass_to_dict(execution.to_result(self._state_revision))

    def provide_input(self, execution_id: str, input_request_id: str, text: str) -> dict:
        execution = self._executions.get(execution_id)
        if not execution:
            raise KeyError(f"Execution {execution_id} not found")
        if execution.status != "input_required" or execution.input_request is None:
            raise RuntimeError("No input request pending")
        if execution.input_request.inputRequestId != input_request_id:
            raise RuntimeError(f"Input request ID mismatch: expected {execution.input_request.inputRequestId}")

        future = execution._input_future
        loop = execution._input_loop
        if future and loop and not future.done():
            loop.call_soon_threadsafe(future.set_result, text)

        # Wait briefly for status to update
        for _ in range(20):
            if execution.status != "input_required":
                break
            time.sleep(0.05)

        return {
            "accepted": True,
            "execution": _dataclass_to_dict(execution.to_state(self._state_revision)),
        }

    def cancel_input(self, execution_id: str, input_request_id: str) -> dict:
        execution = self._executions.get(execution_id)
        if not execution:
            raise KeyError(f"Execution {execution_id} not found")
        if execution.status != "input_required" or execution.input_request is None:
            raise RuntimeError("No input request pending")

        future = execution._input_future
        loop = execution._input_loop
        if future and loop and not future.done():
            loop.call_soon_threadsafe(future.set_exception, InputCancelledError("Input cancelled"))

        # Wait briefly for status to update
        for _ in range(20):
            if execution.is_terminal:
                break
            time.sleep(0.05)

        return {
            "cancelled": True,
            "execution": _dataclass_to_dict(execution.to_state(self._state_revision)),
        }

    def cancel_execution(self, execution_id: str, reason: str | None = None) -> dict:
        execution = self._executions.get(execution_id)
        if not execution:
            raise KeyError(f"Execution {execution_id} not found")
        if execution.is_terminal:
            raise RuntimeError(f"Execution {execution_id} is already terminal")

        # Cancel pending input if any
        if execution._input_future and not execution._input_future.done():
            loop = execution._input_loop
            if loop:
                loop.call_soon_threadsafe(
                    execution._input_future.set_exception,
                    InputCancelledError("Execution cancelled"),
                )

        # Interrupt the worker
        worker = self._get_worker()
        worker.interrupt()

        execution.set_status("cancelled", reason or "Cancelled")
        rev = self._bump_revision()

        return {
            "cancelled": True,
            "execution": _dataclass_to_dict(execution.to_state(rev)),
        }

    # ── Language Intelligence ────────────────────────────────────

    def complete(self, code: str, cursor: int, **kwargs) -> dict:
        self._check_busy("complete")
        worker = self._get_worker()
        result = worker.complete(code, cursor)
        return _dataclass_to_dict(result)

    def inspect(self, code: str, cursor: int, detail: int = 1) -> dict:
        self._check_busy("inspect")
        worker = self._get_worker()
        result = worker.inspect(code, cursor, detail)
        return _dataclass_to_dict(result)

    def hover(self, code: str, cursor: int) -> dict:
        self._check_busy("hover")
        worker = self._get_worker()
        result = worker.hover(code, cursor)
        return _dataclass_to_dict(result)

    # ── Variables ────────────────────────────────────────────────

    def list_variables(self, **kwargs) -> dict:
        self._check_busy("list_variables")
        worker = self._get_worker()
        result = worker.get_variables()
        return _dataclass_to_dict(result)

    def get_variable(self, name: str, path: list[str] | None = None, **kwargs) -> dict:
        self._check_busy("get_variable")
        worker = self._get_worker()
        result = worker.get_variable_detail(name, path)
        return _dataclass_to_dict(result)

    # ── Code Analysis ────────────────────────────────────────────

    def is_complete(self, code: str) -> dict:
        worker = self._get_worker()
        result = worker.is_complete(code)
        return _dataclass_to_dict(result)

    def format_code(self, code: str) -> dict:
        worker = self._get_worker()
        formatted, changed = worker.format_code(code)
        return {"formatted": formatted, "changed": changed}

    # ── History ──────────────────────────────────────────────────

    def query_history(self, n: int = 20, pattern: str | None = None, before: int | None = None) -> dict:
        worker = self._get_worker()
        result = worker.get_history(n=n, pattern=pattern, before=before)
        return _dataclass_to_dict(result)

    # ── Assets ───────────────────────────────────────────────────

    def get_asset_path(self, asset_id: str) -> Path | None:
        """Return the filesystem path for an asset, or None if not found."""
        if "/" in asset_id or ".." in asset_id:
            return None
        full_path = Path(self.assets_dir) / asset_id
        if full_path.exists() and full_path.is_file():
            return full_path
        return None

    # ── Internal ─────────────────────────────────────────────────

    def _check_busy(self, operation: str):
        """Raise if runtime is busy and worker can't safely service introspection."""
        if self._current and not self._current.is_terminal:
            worker = self._get_worker()
            if not worker.supports_busy_introspection():
                raise RuntimeError(f"Runtime busy, cannot service {operation}")

    def shutdown(self):
        with self._lock:
            if self._worker and hasattr(self._worker, "shutdown"):
                self._worker.shutdown()
            self._worker = None
