"""
Subprocess Manager - Manages persistent venv subprocess from the server side.

This module provides SubprocessWorker, which spawns and manages a persistent
Python subprocess running in a venv. The subprocess holds an IPython shell,
so variables persist across executions.

Killing the subprocess releases all resources (including GPU memory).
"""

import asyncio
import json
import os
import sys
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Any
from dataclasses import dataclass, field

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


def get_venv_python(venv_path: str) -> str | None:
    """Get the Python executable path for a venv."""
    venv = Path(venv_path)
    if sys.platform == "win32":
        python_exe = venv / "Scripts" / "python.exe"
    else:
        python_exe = venv / "bin" / "python"
    return str(python_exe) if python_exe.exists() else None


class SubprocessWorker:
    """
    Manages a persistent Python subprocess running in a venv.

    The subprocess runs subprocess_worker.py and communicates via JSON
    over stdin/stdout. Variables persist across executions within the
    subprocess.

    Destroying this worker kills the subprocess, releasing all resources
    including GPU memory.
    """

    def __init__(
        self,
        venv: str,
        cwd: str | None = None,
        assets_dir: str | None = None,
    ):
        self.venv = venv
        self.cwd = cwd
        self.assets_dir = assets_dir

        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._started = False
        self._pid: int | None = None

        # For reading responses
        self._response_queue: dict[int, Any] = {}
        self._request_counter = 0

    def _ensure_started(self):
        """Ensure the subprocess is running."""
        if self._started and self._process and self._process.poll() is None:
            return

        with self._lock:
            # Double-check after acquiring lock
            if self._started and self._process and self._process.poll() is None:
                return

            python_exe = get_venv_python(self.venv)
            if not python_exe:
                raise RuntimeError(f"Could not find Python executable in venv: {self.venv}")

            # Build command
            cmd = [
                python_exe,
                "-m", "mrmd_python.subprocess_worker",
            ]
            if self.cwd:
                cmd.extend(["--cwd", self.cwd])
            if self.assets_dir:
                cmd.extend(["--assets-dir", self.assets_dir])

            # Start subprocess
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=self.cwd,
            )

            # Wait for ready signal
            try:
                ready_line = self._process.stdout.readline()
                ready = json.loads(ready_line)
                if ready.get("type") != "ready":
                    raise RuntimeError(f"Subprocess didn't send ready signal: {ready_line}")
                self._pid = ready.get("pid")
                self._started = True
            except Exception as e:
                self._process.kill()
                self._process = None
                raise RuntimeError(f"Failed to start subprocess: {e}")

    def _send_command(self, command: dict) -> dict:
        """Send a command and wait for response."""
        self._ensure_started()

        with self._lock:
            # Send command
            self._process.stdin.write(json.dumps(command) + "\n")
            self._process.stdin.flush()

            # Read response
            response_line = self._process.stdout.readline()
            if not response_line:
                raise RuntimeError("Subprocess closed unexpectedly")

            return json.loads(response_line)

    def execute(
        self,
        code: str,
        store_history: bool = True,
        exec_id: str | None = None,
    ) -> ExecuteResult:
        """Execute code in the subprocess."""
        response = self._send_command({
            "type": "execute",
            "code": code,
            "storeHistory": store_history,
            "execId": exec_id,
        })

        if response.get("type") == "error":
            return ExecuteResult(
                success=False,
                error=ExecuteError(
                    type="SubprocessError",
                    message=response.get("error", "Unknown error"),
                    traceback=response.get("traceback", "").split("\n") if response.get("traceback") else [],
                ),
            )

        result_data = response.get("result", {})
        return self._parse_execute_result(result_data)

    def execute_streaming(
        self,
        code: str,
        on_output: Callable[[str, str, str], None],
        store_history: bool = True,
        exec_id: str | None = None,
        on_stdin_request: Callable | None = None,
    ) -> ExecuteResult:
        """
        Execute code with streaming output.

        For the subprocess worker, we handle streaming by reading stderr
        in a background thread while the execution runs.
        """
        self._ensure_started()

        # For subprocess streaming, we'll use a simpler approach:
        # Send execute command and poll for output
        with self._lock:
            self._process.stdin.write(json.dumps({
                "type": "execute",
                "code": code,
                "storeHistory": store_history,
                "execId": exec_id,
            }) + "\n")
            self._process.stdin.flush()

            accumulated_stdout = ""
            accumulated_stderr = ""

            # Read the response (the subprocess will have captured all output)
            response_line = self._process.stdout.readline()
            if not response_line:
                raise RuntimeError("Subprocess closed unexpectedly")

            response = json.loads(response_line)

        if response.get("type") == "error":
            return ExecuteResult(
                success=False,
                error=ExecuteError(
                    type="SubprocessError",
                    message=response.get("error", "Unknown error"),
                ),
            )

        result_data = response.get("result", {})
        result = self._parse_execute_result(result_data)

        # Send final output via callback
        if result.stdout:
            on_output("stdout", result.stdout, result.stdout)
        if result.stderr:
            on_output("stderr", result.stderr, result.stderr)

        return result

    def complete(self, code: str, cursor_pos: int) -> CompleteResult:
        """Get completions at cursor position."""
        response = self._send_command({
            "type": "complete",
            "code": code,
            "cursor": cursor_pos,
        })

        if response.get("type") == "error":
            return CompleteResult(cursorStart=cursor_pos, cursorEnd=cursor_pos)

        result = response.get("result", {})
        matches = []
        for m in result.get("matches", []):
            matches.append(CompletionItem(
                label=m.get("label", ""),
                kind=m.get("kind", "variable"),
                type=m.get("type"),
            ))

        return CompleteResult(
            matches=matches,
            cursorStart=result.get("cursorStart", cursor_pos),
            cursorEnd=result.get("cursorEnd", cursor_pos),
            source=result.get("source", "runtime"),
        )

    def inspect(self, code: str, cursor_pos: int, detail: int = 1) -> InspectResult:
        """Get detailed info about symbol at cursor."""
        response = self._send_command({
            "type": "inspect",
            "code": code,
            "cursor": cursor_pos,
            "detail": detail,
        })

        if response.get("type") == "error":
            return InspectResult(found=False)

        result = response.get("result", {})
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
        response = self._send_command({
            "type": "hover",
            "code": code,
            "cursor": cursor_pos,
        })

        if response.get("type") == "error":
            return HoverResult(found=False)

        result = response.get("result", {})
        return HoverResult(
            found=result.get("found", False),
            name=result.get("name"),
            type=result.get("type"),
            value=result.get("value"),
            signature=result.get("signature"),
        )

    def get_variables(self) -> VariablesResult:
        """Get user variables."""
        response = self._send_command({"type": "variables"})

        if response.get("type") == "error":
            return VariablesResult(variables=[], count=0, truncated=False)

        result = response.get("result", {})
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

    def get_variable_detail(self, name: str, path: list[str] | None = None) -> VariableDetail:
        """Get detailed info about a variable."""
        # For now, just return basic info from variables
        vars_result = self.get_variables()
        for v in vars_result.variables:
            if v.name == name:
                return VariableDetail(
                    name=v.name,
                    type=v.type,
                    value=v.value,
                    size=v.size,
                    expandable=v.expandable,
                    shape=v.shape,
                    dtype=v.dtype,
                    length=v.length,
                )
        return VariableDetail(name=name, type="unknown", value="<not found>")

    def is_complete(self, code: str) -> IsCompleteResult:
        """Check if code is a complete statement."""
        response = self._send_command({
            "type": "is_complete",
            "code": code,
        })

        if response.get("type") == "error":
            return IsCompleteResult(status="unknown")

        result = response.get("result", {})
        return IsCompleteResult(
            status=result.get("status", "unknown"),
            indent=result.get("indent", ""),
        )

    def format_code(self, code: str) -> tuple[str, bool]:
        """Format code using black (not implemented in subprocess yet)."""
        # Could implement this in subprocess_worker if needed
        return code, False

    def reset(self):
        """Reset the namespace."""
        self._send_command({"type": "reset"})

    def get_info(self) -> dict:
        """Get info about this worker."""
        python_exe = get_venv_python(self.venv)
        return {
            "python_executable": python_exe,
            "venv": self.venv,
            "cwd": self.cwd,
            "pid": self._pid,
            "running": self._process is not None and self._process.poll() is None,
        }

    def shutdown(self):
        """Shutdown the subprocess, using force kill to ensure GPU memory release.

        This MUST kill the process to release GPU memory for vLLM workloads.
        We use kill() (SIGKILL) directly to ensure the process dies immediately.
        """
        if self._process is None:
            return

        pid = self._pid  # Save for logging

        # Force kill immediately - don't bother with graceful shutdown
        # GPU memory release requires the process to actually die
        try:
            self._process.kill()  # SIGKILL
        except Exception:
            pass

        try:
            self._process.wait(timeout=2.0)
        except Exception:
            pass

        # Reset state
        self._process = None
        self._started = False
        self._pid = None

    def kill(self):
        """Force kill the subprocess."""
        if self._process:
            try:
                self._process.kill()
                self._process.wait(timeout=1.0)
            except Exception:
                pass
            finally:
                self._process = None
                self._started = False
                self._pid = None

    def interrupt(self) -> bool:
        """Send SIGINT to the subprocess to raise KeyboardInterrupt.

        Returns True if signal was sent successfully.
        """
        if self._process and self._process.poll() is None:
            try:
                import signal
                self._process.send_signal(signal.SIGINT)
                return True
            except Exception:
                pass
        return False

    def is_alive(self) -> bool:
        """Check if subprocess is still running."""
        return self._process is not None and self._process.poll() is None

    def _parse_execute_result(self, data: dict) -> ExecuteResult:
        """Parse execute result from subprocess response."""
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

    def __del__(self):
        """Cleanup on garbage collection."""
        self.shutdown()
