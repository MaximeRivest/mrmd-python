"""Kernel process manager for mrmd-python.

The parent server keeps HTTP handling in one process and delegates all Python
runtime state to a persistent child kernel process. User code runs on the child
process main thread, which gives us normal Python signal/KeyboardInterrupt
semantics on POSIX and a clean restart fallback everywhere else.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from .types import (
    Asset,
    CompleteResult,
    CompletionItem,
    DisplayData,
    ExecuteError,
    ExecuteResult,
    HistoryEntry,
    HistoryResult,
    HoverResult,
    InspectResult,
    IsCompleteResult,
    Variable,
    VariableDetail,
    VariablesResult,
)


def get_venv_python(venv_path: str) -> str | None:
    """Get the Python executable path for a venv or prefix."""
    venv = Path(venv_path)
    if sys.platform == "win32":
        candidates = [venv / "Scripts" / "python.exe"]
    else:
        candidates = [venv / "bin" / "python", venv / "bin" / "python3"]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


class SubprocessWorker:
    """Manage a persistent child kernel process."""

    READY_TIMEOUT = 10.0
    INTERRUPT_GRACE_PERIOD = 1.5
    TERMINATE_TIMEOUT = 2.0

    def __init__(self, venv: str, cwd: str | None = None, assets_dir: str | None = None):
        self.venv = venv
        self.cwd = cwd
        self.assets_dir = assets_dir

        self._process: subprocess.Popen[str] | None = None
        self._pid: int | None = None
        self._started = False
        self._generation = 0

        self._state_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._request_counter = 0
        self._ready_event = threading.Event()
        self._start_error: str | None = None
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._exit_error: dict[str, Any] | None = None
        self._active_execution_id: int | None = None
        self._active_execution_done = threading.Event()
        self._active_execution_done.set()

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._started and self._process and self._process.poll() is None:
            return

        with self._state_lock:
            if self._started and self._process and self._process.poll() is None:
                return

            python_exe = get_venv_python(self.venv)
            if not python_exe:
                raise RuntimeError(f"Could not find Python executable in venv: {self.venv}")

            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")

            cmd = [python_exe, "-m", "mrmd_python.kernel_process"]
            if self.cwd:
                cmd.extend(["--cwd", self.cwd])
            if self.assets_dir:
                cmd.extend(["--assets-dir", self.assets_dir])

            popen_kwargs: dict[str, Any] = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "bufsize": 1,
                "cwd": self.cwd,
                "env": env,
            }
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if creationflags:
                    popen_kwargs["creationflags"] = creationflags
            else:
                popen_kwargs["start_new_session"] = True

            self._generation += 1
            generation = self._generation
            self._ready_event = threading.Event()
            self._start_error = None
            self._exit_error = None
            self._active_execution_id = None
            self._active_execution_done.set()

            self._process = subprocess.Popen(cmd, **popen_kwargs)
            self._pid = self._process.pid
            self._started = False

            threading.Thread(
                target=self._reader_loop,
                args=(generation, self._process),
                name=f"mrmd-python-kernel-reader-{generation}",
                daemon=True,
            ).start()
            threading.Thread(
                target=self._stderr_loop,
                args=(generation, self._process),
                name=f"mrmd-python-kernel-stderr-{generation}",
                daemon=True,
            ).start()

        if not self._ready_event.wait(self.READY_TIMEOUT):
            self._terminate_process_tree(force=True)
            stderr_text = "".join(self._stderr_tail).strip()
            detail = self._start_error or stderr_text or "kernel did not become ready"
            raise RuntimeError(f"Failed to start kernel process: {detail}")

        if not self._started:
            stderr_text = "".join(self._stderr_tail).strip()
            detail = self._start_error or stderr_text or "kernel start failed"
            raise RuntimeError(f"Failed to start kernel process: {detail}")

    def _reader_loop(self, generation: int, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        try:
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    with self._state_lock:
                        if generation == self._generation:
                            self._stderr_tail.append(f"[protocol] {line}\n")
                    continue

                msg_type = message.get("type")
                if msg_type == "ready":
                    with self._state_lock:
                        if generation != self._generation:
                            continue
                        self._pid = message.get("pid", self._pid)
                        self._started = True
                        self._ready_event.set()
                    continue

                if msg_type in {"response", "event"}:
                    request_id = message.get("id")
                    with self._state_lock:
                        pending = self._pending.get(request_id)
                    if pending is not None:
                        pending.put(message)
                    continue

                if msg_type == "protocol_error":
                    with self._state_lock:
                        if generation == self._generation:
                            self._stderr_tail.append(json.dumps(message) + "\n")
                    continue
        finally:
            with self._state_lock:
                if generation != self._generation:
                    return
                if not self._ready_event.is_set():
                    self._start_error = "kernel exited before sending ready"
                    self._ready_event.set()
            self._handle_process_exit(generation)

    def _stderr_loop(self, generation: int, process: subprocess.Popen[str]) -> None:
        stderr = process.stderr
        if stderr is None:
            return
        for line in stderr:
            with self._state_lock:
                if generation != self._generation:
                    return
                self._stderr_tail.append(line)

    def _handle_process_exit(self, generation: int) -> None:
        with self._state_lock:
            if generation != self._generation:
                return

            error_payload = self._exit_error or {
                "type": "KernelExited",
                "message": "Kernel process exited unexpectedly",
                "traceback": [],
            }
            pending = list(self._pending.values())
            self._pending.clear()
            self._process = None
            self._pid = None
            self._started = False
            self._ready_event.set()
            self._active_execution_id = None
            self._active_execution_done.set()
            self._exit_error = None

        for pending_queue in pending:
            pending_queue.put({"type": "process_exit", "error": error_payload})

    def _send_message(self, message: dict[str, Any]) -> None:
        self._ensure_started()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Kernel process is not available")

        line = json.dumps(message, ensure_ascii=False)
        with self._write_lock:
            process.stdin.write(line + "\n")
            process.stdin.flush()

    def _next_request_id(self) -> int:
        with self._state_lock:
            self._request_counter += 1
            return self._request_counter

    def _create_pending_queue(self, request_id: int) -> queue.Queue[dict[str, Any]]:
        q: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._state_lock:
            self._pending[request_id] = q
        return q

    def _remove_pending_queue(self, request_id: int) -> None:
        with self._state_lock:
            self._pending.pop(request_id, None)

    def _begin_execution(self, request_id: int) -> None:
        with self._state_lock:
            self._active_execution_id = request_id
            self._active_execution_done.clear()

    def _finish_execution(self, request_id: int) -> None:
        with self._state_lock:
            if self._active_execution_id == request_id:
                self._active_execution_id = None
                self._active_execution_done.set()

    def _request(self, command: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        request_id = self._next_request_id()
        pending = self._create_pending_queue(request_id)
        try:
            self._send_message({"id": request_id, **command})
            return pending.get(timeout=timeout)
        finally:
            self._remove_pending_queue(request_id)

    def _execution_request(self, command: dict[str, Any]) -> tuple[int, queue.Queue[dict[str, Any]]]:
        request_id = self._next_request_id()
        pending = self._create_pending_queue(request_id)
        self._begin_execution(request_id)
        try:
            self._send_message({"id": request_id, **command})
        except Exception:
            self._finish_execution(request_id)
            self._remove_pending_queue(request_id)
            raise
        return request_id, pending

    def _response_error(self, response: dict[str, Any], default_type: str = "KernelError") -> ExecuteError:
        err = response.get("error") or {}
        return ExecuteError(
            type=err.get("type", default_type),
            message=err.get("message", "Kernel error"),
            traceback=err.get("traceback", []),
            line=err.get("line"),
            column=err.get("column"),
        )

    def _execute_error_result(self, response: dict[str, Any], default_type: str = "KernelError") -> ExecuteResult:
        return ExecuteResult(success=False, error=self._response_error(response, default_type=default_type))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        code: str,
        store_history: bool = True,
        exec_id: str | None = None,
        allow_stdin: bool = False,
    ) -> ExecuteResult:
        request_id, pending = self._execution_request({
            "type": "execute",
            "code": code,
            "storeHistory": store_history,
            "execId": exec_id,
            "allowStdin": allow_stdin,
        })
        try:
            response = pending.get()
            if response.get("type") == "process_exit":
                return self._execute_error_result(response, default_type="KernelExited")
            if response.get("error"):
                return self._execute_error_result(response)
            return self._parse_execute_result(response.get("result", {}))
        finally:
            self._finish_execution(request_id)
            self._remove_pending_queue(request_id)

    def execute_streaming(
        self,
        code: str,
        on_output: Callable[[str, str, str], None],
        store_history: bool = True,
        exec_id: str | None = None,
        on_stdin_request: Callable | None = None,
        allow_stdin: bool = True,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> ExecuteResult:
        request_id, pending = self._execution_request({
            "type": "execute_stream",
            "code": code,
            "storeHistory": store_history,
            "execId": exec_id,
            "allowStdin": allow_stdin,
        })
        accumulated_stdout = ""
        accumulated_stderr = ""

        try:
            while True:
                response = pending.get()
                response_type = response.get("type")

                if response_type == "process_exit":
                    return self._execute_error_result(response, default_type="KernelExited")

                if response_type == "event":
                    event_type = response.get("event")
                    data = response.get("data", {})

                    if event_type in {"stdout", "stderr"}:
                        content = data.get("content", "")
                        if event_type == "stdout":
                            accumulated_stdout += content
                            on_output("stdout", content, accumulated_stdout)
                        else:
                            accumulated_stderr += content
                            on_output("stderr", content, accumulated_stderr)
                        continue

                    if event_type == "stdin_request" and on_stdin_request is not None:
                        try:
                            text = on_stdin_request(SimpleNamespace(**data))
                        except Exception:
                            self._send_message({
                                "type": "input_cancel",
                                "execId": data.get("execId", exec_id or ""),
                            })
                        else:
                            self._send_message({
                                "type": "input_response",
                                "execId": data.get("execId", exec_id or ""),
                                "text": text,
                            })
                        continue

                    if on_event and event_type:
                        on_event(event_type, data)
                    continue

                if response.get("error"):
                    return self._execute_error_result(response)

                return self._parse_execute_result(response.get("result", {}))
        finally:
            self._finish_execution(request_id)
            self._remove_pending_queue(request_id)

    def complete(self, code: str, cursor_pos: int) -> CompleteResult:
        response = self._request({"type": "complete", "code": code, "cursor": cursor_pos})
        if response.get("type") == "process_exit" or response.get("error"):
            return CompleteResult(cursorStart=cursor_pos, cursorEnd=cursor_pos)

        result = response.get("result", {})
        matches = [
            CompletionItem(
                label=item.get("label", ""),
                insertText=item.get("insertText"),
                kind=item.get("kind", "variable"),
                detail=item.get("detail"),
                documentation=item.get("documentation"),
                valuePreview=item.get("valuePreview"),
                type=item.get("type"),
                sortText=item.get("sortText"),
            )
            for item in result.get("matches", [])
        ]
        return CompleteResult(
            matches=matches,
            cursorStart=result.get("cursorStart", cursor_pos),
            cursorEnd=result.get("cursorEnd", cursor_pos),
            source=result.get("source", "runtime"),
        )

    def inspect(self, code: str, cursor_pos: int, detail: int = 1) -> InspectResult:
        response = self._request({
            "type": "inspect",
            "code": code,
            "cursor": cursor_pos,
            "detail": detail,
        })
        if response.get("type") == "process_exit" or response.get("error"):
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
            value=result.get("value"),
            children=result.get("children"),
        )

    def hover(self, code: str, cursor_pos: int) -> HoverResult:
        response = self._request({"type": "hover", "code": code, "cursor": cursor_pos})
        if response.get("type") == "process_exit" or response.get("error"):
            return HoverResult(found=False)

        result = response.get("result", {})
        return HoverResult(
            found=result.get("found", False),
            name=result.get("name"),
            type=result.get("type"),
            value=result.get("value"),
            signature=result.get("signature"),
            docstring=result.get("docstring"),
        )

    def get_variables(self) -> VariablesResult:
        response = self._request({"type": "variables"})
        if response.get("type") == "process_exit" or response.get("error"):
            return VariablesResult(variables=[], count=0, truncated=False)

        result = response.get("result", {})
        variables = [
            Variable(
                name=item.get("name", ""),
                type=item.get("type", ""),
                value=item.get("value", ""),
                size=item.get("size"),
                expandable=item.get("expandable", False),
                shape=item.get("shape"),
                dtype=item.get("dtype"),
                length=item.get("length"),
                keys=item.get("keys"),
            )
            for item in result.get("variables", [])
        ]
        return VariablesResult(
            variables=variables,
            count=result.get("count", len(variables)),
            truncated=result.get("truncated", False),
            warnings=result.get("warnings", []),
        )

    def get_variable_detail(self, name: str, path: list[str] | None = None) -> VariableDetail:
        response = self._request({"type": "variable_detail", "name": name, "path": path or []})
        if response.get("type") == "process_exit" or response.get("error"):
            return VariableDetail(name=name, type="unknown", value="<not found>")

        result = response.get("result", {})
        children = [
            Variable(
                name=item.get("name", ""),
                type=item.get("type", ""),
                value=item.get("value", ""),
                size=item.get("size"),
                expandable=item.get("expandable", False),
                shape=item.get("shape"),
                dtype=item.get("dtype"),
                length=item.get("length"),
                keys=item.get("keys"),
            )
            for item in result.get("children", []) or []
        ]
        return VariableDetail(
            name=result.get("name", name),
            type=result.get("type", "unknown"),
            value=result.get("value", ""),
            size=result.get("size"),
            expandable=result.get("expandable", False),
            shape=result.get("shape"),
            dtype=result.get("dtype"),
            length=result.get("length"),
            keys=result.get("keys"),
            fullValue=result.get("fullValue"),
            children=children or None,
            methods=result.get("methods"),
            attributes=result.get("attributes"),
        )

    def is_complete(self, code: str) -> IsCompleteResult:
        response = self._request({"type": "is_complete", "code": code})
        if response.get("type") == "process_exit" or response.get("error"):
            return IsCompleteResult(status="unknown")

        result = response.get("result", {})
        return IsCompleteResult(status=result.get("status", "unknown"), indent=result.get("indent", ""))

    def format_code(self, code: str) -> tuple[str, bool]:
        response = self._request({"type": "format", "code": code})
        if response.get("type") == "process_exit" or response.get("error"):
            return code, False

        result = response.get("result", {})
        return result.get("formatted", code), result.get("changed", False)

    def get_history(self, n: int = 20, pattern: str | None = None, before: int | None = None) -> HistoryResult:
        response = self._request({"type": "history", "n": n, "pattern": pattern, "before": before})
        if response.get("type") == "process_exit" or response.get("error"):
            return HistoryResult(entries=[], hasMore=False)

        result = response.get("result", {})
        entries = [
            HistoryEntry(historyIndex=item.get("historyIndex", 0), code=item.get("code", ""))
            for item in result.get("entries", [])
        ]
        return HistoryResult(entries=entries, hasMore=result.get("hasMore", False))

    def reset(self) -> None:
        self._request({"type": "reset"})

    def clear_history(self) -> bool:
        response = self._request({"type": "clear_history"})
        if response.get("type") == "process_exit" or response.get("error"):
            return False
        return bool(response.get("result", {}).get("success", False))

    def supports_busy_introspection(self) -> bool:
        """Whether this worker can safely answer best-effort introspection while busy."""
        return True

    def get_info(self) -> dict[str, Any]:
        python_exe = get_venv_python(self.venv)
        return {
            "python_executable": python_exe,
            "venv": self.venv,
            "cwd": self.cwd,
            "pid": self._pid,
            "running": self.is_alive(),
        }

    def shutdown(self) -> None:
        self._terminate_process_tree(force=True)

    def kill(self) -> None:
        self._terminate_process_tree(force=True)

    def interrupt(self) -> bool:
        process = self._process
        if process is None or process.poll() is not None:
            return False

        with self._state_lock:
            active_execution = self._active_execution_id

        if active_execution is None:
            return False

        interrupted = True
        try:
            response = self._request({"type": "interrupt"}, timeout=1.0)
            if response.get("type") != "process_exit":
                interrupted = bool(response.get("result", {}).get("interrupted", True))
        except Exception:
            interrupted = True

        self._send_soft_interrupt(process)

        if self._active_execution_done.wait(timeout=self.INTERRUPT_GRACE_PERIOD):
            return interrupted

        with self._state_lock:
            self._exit_error = {
                "type": "Interrupted",
                "message": "Execution interrupted; kernel restarted",
                "traceback": [],
            }
        self._terminate_process_tree(force=False)
        return True

    def _send_soft_interrupt(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            try:
                ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
                if ctrl_break is not None:
                    process.send_signal(ctrl_break)
            except Exception:
                pass
            return

        try:
            os.killpg(process.pid, signal.SIGINT)
        except Exception:
            try:
                process.send_signal(signal.SIGINT)
            except Exception:
                pass

    def _terminate_process_tree(self, force: bool) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is not None:
            self._handle_process_exit(self._generation)
            return

        if os.name == "nt":
            if force:
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                        check=False,
                    )
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            else:
                try:
                    process.terminate()
                except Exception:
                    pass
        else:
            sig = signal.SIGKILL if force else signal.SIGTERM
            try:
                os.killpg(process.pid, sig)
            except Exception:
                try:
                    process.send_signal(sig)
                except Exception:
                    pass

        try:
            process.wait(timeout=self.TERMINATE_TIMEOUT)
        except Exception:
            if not force:
                self._terminate_process_tree(force=True)
                return

        self._handle_process_exit(self._generation)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_execute_result(self, data: dict[str, Any]) -> ExecuteResult:
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

        assets = [
            Asset(
                id=item.get("id", ""),
                url=item.get("url", ""),
                mimeType=item.get("mimeType", ""),
                assetType=item.get("assetType", "file"),
                size=item.get("size", 0),
            )
            for item in data.get("assets", [])
        ]

        display_data = [
            DisplayData(data=item.get("data", {}), metadata=item.get("metadata", {}))
            for item in data.get("displayData", [])
        ]

        return ExecuteResult(
            success=data.get("success", False),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            result=data.get("result"),
            error=error,
            displayData=display_data,
            assets=assets,
            executionCount=data.get("executionCount", 0),
            stateRevision=data.get("stateRevision", 0),
            duration=data.get("duration"),
            imports=data.get("imports", []),
            warnings=data.get("warnings", []),
        )

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
