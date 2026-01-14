"""
Runtime Client - HTTP client for communicating with daemon runtimes.

This module provides DaemonRuntimeClient which connects to independent
daemon runtimes via HTTP. It has the same interface as SubprocessWorker
but communicates over the network instead of stdin/stdout.

The client can:
- Spawn new daemon runtimes if they don't exist
- Connect to existing daemon runtimes
- Execute code, get completions, inspect variables, etc.
- Survive parent process death (the daemon keeps running)
"""

import httpx
import json
import signal
from typing import Callable, Optional
from dataclasses import dataclass

from .types import (
    ExecuteResult,
    ExecuteError,
    Asset,
    DisplayData,
    CompleteResult,
    CompletionItem,
    InspectResult,
    HoverResult,
    VariablesResult,
    Variable,
    VariableDetail,
    IsCompleteResult,
)
from .runtime_daemon import (
    spawn_daemon,
    kill_runtime,
    is_runtime_alive,
    read_runtime_info,
    list_runtimes,
)


class DaemonRuntimeClient:
    """
    Client for communicating with an independent daemon runtime.

    This client connects to a daemon runtime over HTTP. The daemon
    is a fully independent process that survives if this client or
    its parent dies.

    Key features:
    - Spawns daemon on first use if not already running
    - Variables persist across client reconnections
    - Killing the daemon releases all GPU memory
    - Registry stored in ~/.mrmd/runtimes/
    """

    def __init__(
        self,
        runtime_id: str,
        venv: Optional[str] = None,
        cwd: Optional[str] = None,
        assets_dir: Optional[str] = None,
        auto_spawn: bool = True,
    ):
        """
        Initialize the client.

        Args:
            runtime_id: Unique identifier for the runtime
            venv: Virtual environment path (auto-detected if not specified)
            cwd: Working directory for the runtime
            assets_dir: Directory for saving assets
            auto_spawn: If True, spawn daemon if not already running
        """
        self.runtime_id = runtime_id
        self.venv = venv
        self.cwd = cwd
        self.assets_dir = assets_dir
        self.auto_spawn = auto_spawn

        self._info: Optional[dict] = None
        self._client: Optional[httpx.Client] = None

    def _ensure_connected(self):
        """Ensure we're connected to the daemon runtime."""
        # Check if we have a valid connection
        if self._client and self._info:
            # Verify runtime is still alive
            if is_runtime_alive(self.runtime_id):
                return

        # Try to read existing runtime info
        self._info = read_runtime_info(self.runtime_id)

        if self._info and is_runtime_alive(self.runtime_id):
            # Runtime exists, connect to it
            self._client = httpx.Client(
                base_url=self._info["url"],
                timeout=300.0,  # 5 minute timeout for long executions
            )
            return

        # Runtime doesn't exist or is dead
        if not self.auto_spawn:
            raise RuntimeError(f"Runtime {self.runtime_id} is not running")

        # Spawn new daemon
        self._info = spawn_daemon(
            runtime_id=self.runtime_id,
            venv=self.venv,
            cwd=self.cwd,
            assets_dir=self.assets_dir,
        )

        # Connect to it
        self._client = httpx.Client(
            base_url=self._info["url"],
            timeout=300.0,
        )

    def _post(self, endpoint: str, data: dict) -> dict:
        """Make a POST request to the daemon."""
        self._ensure_connected()
        response = self._client.post(endpoint, json=data)
        response.raise_for_status()
        return response.json()

    def _get(self, endpoint: str) -> dict:
        """Make a GET request to the daemon."""
        self._ensure_connected()
        response = self._client.get(endpoint)
        response.raise_for_status()
        return response.json()

    def execute(
        self,
        code: str,
        store_history: bool = True,
        exec_id: Optional[str] = None,
    ) -> ExecuteResult:
        """Execute code in the daemon runtime."""
        data = {
            "code": code,
            "storeHistory": store_history,
            "session": "default",  # Daemon uses default session
        }
        if exec_id:
            data["execId"] = exec_id

        result = self._post("/execute", data)
        return self._parse_execute_result(result)

    def execute_streaming(
        self,
        code: str,
        on_output: Callable[[str, str, str], None],
        store_history: bool = True,
        exec_id: Optional[str] = None,
        on_stdin_request: Optional[Callable] = None,
    ) -> ExecuteResult:
        """
        Execute code with streaming output.

        For daemon runtimes, we use SSE streaming via HTTP.
        """
        self._ensure_connected()

        data = {
            "code": code,
            "storeHistory": store_history,
            "session": "default",
        }
        if exec_id:
            data["execId"] = exec_id

        accumulated = {"stdout": "", "stderr": ""}
        final_result = None

        # Use httpx streaming for SSE
        with self._client.stream("POST", "/execute/stream", json=data) as response:
            for line in response.iter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    event_data = json.loads(line[5:])

                    if event_type in ("stdout", "stderr"):
                        content = event_data.get("content", "")
                        accumulated[event_type] = event_data.get("accumulated", accumulated[event_type] + content)
                        on_output(event_type, content, accumulated[event_type])
                    elif event_type == "result":
                        final_result = self._parse_execute_result(event_data)
                    elif event_type == "error":
                        final_result = ExecuteResult(
                            success=False,
                            error=ExecuteError(
                                type=event_data.get("type", "Error"),
                                message=event_data.get("message", ""),
                                traceback=event_data.get("traceback", []),
                            ),
                        )
                    elif event_type == "stdin_request":
                        if on_stdin_request:
                            from .types import StdinRequest
                            response_text = on_stdin_request(StdinRequest(
                                prompt=event_data.get("prompt", ""),
                                password=event_data.get("password", False),
                                execId=event_data.get("execId", ""),
                            ))
                            # Send input back
                            self._post("/input", {
                                "exec_id": event_data.get("execId", ""),
                                "text": response_text,
                            })

        return final_result or ExecuteResult(success=False)

    def complete(self, code: str, cursor_pos: int) -> CompleteResult:
        """Get completions at cursor position."""
        result = self._post("/complete", {
            "code": code,
            "cursor": cursor_pos,
            "session": "default",
        })

        matches = []
        for m in result.get("matches", []):
            if isinstance(m, dict):
                matches.append(CompletionItem(
                    label=m.get("label", ""),
                    kind=m.get("kind", "variable"),
                    type=m.get("type"),
                ))
            else:
                matches.append(CompletionItem(label=str(m), kind="variable"))

        return CompleteResult(
            matches=matches,
            cursorStart=result.get("cursorStart", cursor_pos),
            cursorEnd=result.get("cursorEnd", cursor_pos),
            source=result.get("source", "runtime"),
        )

    def inspect(self, code: str, cursor_pos: int, detail: int = 1) -> InspectResult:
        """Get detailed info about symbol at cursor."""
        result = self._post("/inspect", {
            "code": code,
            "cursor": cursor_pos,
            "detail": detail,
            "session": "default",
        })

        return InspectResult(
            found=result.get("found", False),
            source=result.get("source", "runtime"),
            name=result.get("name"),
            kind=result.get("kind"),
            type=result.get("type"),
            signature=result.get("signature"),
            docstring=result.get("docstring"),
            sourceCode=result.get("sourceCode"),
            file=result.get("file"),
            line=result.get("line"),
        )

    def hover(self, code: str, cursor_pos: int) -> HoverResult:
        """Get hover tooltip for symbol."""
        result = self._post("/hover", {
            "code": code,
            "cursor": cursor_pos,
            "session": "default",
        })

        return HoverResult(
            found=result.get("found", False),
            name=result.get("name"),
            type=result.get("type"),
            value=result.get("value"),
            signature=result.get("signature"),
        )

    def get_variables(self) -> VariablesResult:
        """Get user variables."""
        result = self._post("/variables", {"session": "default"})

        variables = []
        for v in result.get("variables", []):
            variables.append(Variable(
                name=v.get("name", ""),
                type=v.get("type", ""),
                value=v.get("value", ""),
                size=v.get("size"),
                expandable=v.get("expandable", False),
                shape=v.get("shape"),
                dtype=v.get("dtype"),
                length=v.get("length"),
            ))

        return VariablesResult(
            variables=variables,
            count=result.get("count", len(variables)),
            truncated=result.get("truncated", False),
        )

    def get_variable_detail(self, name: str, path: Optional[list[str]] = None) -> VariableDetail:
        """Get detailed info about a variable."""
        result = self._post(f"/variables/{name}", {
            "session": "default",
            "path": path,
        })

        children = None
        if result.get("children"):
            children = []
            for c in result["children"]:
                children.append(Variable(
                    name=c.get("name", ""),
                    type=c.get("type", ""),
                    value=c.get("value", ""),
                    size=c.get("size"),
                    expandable=c.get("expandable", False),
                ))

        return VariableDetail(
            name=result.get("name", name),
            type=result.get("type", "unknown"),
            value=result.get("value", ""),
            size=result.get("size"),
            expandable=result.get("expandable", False),
            shape=result.get("shape"),
            dtype=result.get("dtype"),
            length=result.get("length"),
            fullValue=result.get("fullValue"),
            children=children,
        )

    def is_complete(self, code: str) -> IsCompleteResult:
        """Check if code is a complete statement."""
        result = self._post("/is_complete", {
            "code": code,
            "session": "default",
        })

        return IsCompleteResult(
            status=result.get("status", "unknown"),
            indent=result.get("indent", ""),
        )

    def format_code(self, code: str) -> tuple[str, bool]:
        """Format code using black."""
        result = self._post("/format", {
            "code": code,
            "session": "default",
        })

        return result.get("formatted", code), result.get("changed", False)

    def reset(self):
        """Reset the runtime namespace."""
        self._ensure_connected()
        # Use the session reset endpoint
        self._client.post("/sessions/default/reset")

    def get_info(self) -> dict:
        """Get info about the daemon runtime."""
        if not self._info:
            self._info = read_runtime_info(self.runtime_id)
        return {
            "runtime_id": self.runtime_id,
            "pid": self._info.get("pid") if self._info else None,
            "port": self._info.get("port") if self._info else None,
            "url": self._info.get("url") if self._info else None,
            "venv": self._info.get("venv") if self._info else self.venv,
            "cwd": self._info.get("cwd") if self._info else self.cwd,
            "alive": is_runtime_alive(self.runtime_id),
        }

    def shutdown(self):
        """
        Shutdown (kill) the daemon runtime.

        This releases all GPU memory.
        """
        if self._client:
            self._client.close()
            self._client = None

        kill_runtime(self.runtime_id)
        self._info = None

    def kill(self):
        """Alias for shutdown - force kill the daemon."""
        self.shutdown()

    def is_alive(self) -> bool:
        """Check if the daemon runtime is still running."""
        return is_runtime_alive(self.runtime_id)

    def _parse_execute_result(self, data: dict) -> ExecuteResult:
        """Parse execute result from HTTP response."""
        error = None
        if data.get("error"):
            err = data["error"]
            error = ExecuteError(
                type=err.get("type", "Error"),
                message=err.get("message", ""),
                traceback=err.get("traceback", []),
                line=err.get("line"),
                column=err.get("column"),
            )

        assets = []
        for a in data.get("assets", []):
            assets.append(Asset(
                path=a.get("path", ""),
                url=a.get("url", ""),
                mimeType=a.get("mimeType", ""),
                assetType=a.get("assetType", ""),
                size=a.get("size", 0),
            ))

        display_data = []
        for d in data.get("displayData", []):
            display_data.append(DisplayData(
                data=d.get("data", {}),
                metadata=d.get("metadata", {}),
            ))

        return ExecuteResult(
            success=data.get("success", False),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            result=data.get("result"),
            error=error,
            displayData=display_data,
            assets=assets,
            executionCount=data.get("executionCount", 0),
            duration=data.get("duration"),
        )

    def _format_exception(self, exc: Exception) -> ExecuteError:
        """Format exception to ExecuteError."""
        import traceback
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        return ExecuteError(
            type=type(exc).__name__,
            message=str(exc),
            traceback=tb_lines,
        )

    def __del__(self):
        """Cleanup client connection (but NOT the daemon)."""
        if self._client:
            self._client.close()
