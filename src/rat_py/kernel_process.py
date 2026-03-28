"""Persistent Python kernel process for rat-py.

This module runs a dedicated runtime process whose *main thread* executes user
code. Commands arrive over stdin as line-delimited JSON and responses/events are
sent back over a duplicated copy of stdout so the control channel stays alive
while execution temporarily redirects fd 1/2 for terminal-style streaming.

Why this exists:
- Python signal delivery is main-thread oriented
- in-process thread interrupts are unreliable for blocking C calls
- a dedicated kernel process gives us normal Python signal/KeyboardInterrupt
  semantics on POSIX and a clean restart fallback everywhere else

Busy-time introspection:
- while code is executing, the reader thread serves snapshot-backed requests
  (`/complete`, `/hover`, `/inspect`, `/variables`, `/variables/{name}`,
  `/history`) from the last committed runtime snapshot
- `/is_complete` and `/format` are also handled out-of-band during execution
- this keeps the external MRP interface unchanged while making the degraded,
  snapshot-based behavior an internal implementation detail
"""

from __future__ import annotations

import _thread
import inspect
import json
import os
import queue
import signal
import sys
import threading
import traceback
from dataclasses import asdict, is_dataclass
from typing import Any

from .types import ExecuteError, ExecuteResult, InputCancelledError, StdinRequest
from .worker import IPythonWorker


SNAPSHOT_MAX_DEPTH = 3
SNAPSHOT_MAX_CHILDREN = 100
STALENESS_WARNING = {
    "code": "stale_during_execution",
    "message": "Result is based on the last committed runtime snapshot while execution is still running.",
}


def _to_jsonable(obj: Any) -> Any:
    """Convert dataclasses/nested containers to JSON-safe values."""
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


class ProtocolWriter:
    """Thread-safe line-oriented JSON writer using a dedicated control fd."""

    def __init__(self, fd: int):
        self._lock = threading.Lock()
        self._file = os.fdopen(fd, "w", buffering=1, encoding="utf-8", errors="replace")

    def send(self, message: dict[str, Any]) -> None:
        line = json.dumps(_to_jsonable(message), ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()


class InputMailbox:
    """Single pending stdin request mailbox."""

    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._exec_id: str | None = None
        self._text: str | None = None
        self._cancelled = False
        self._cancel_message = "Input cancelled by user"

    def begin(self, exec_id: str) -> None:
        with self._lock:
            self._exec_id = exec_id
            self._text = None
            self._cancelled = False
            self._cancel_message = "Input cancelled by user"
            self._event = threading.Event()

    def provide(self, exec_id: str, text: str) -> bool:
        with self._lock:
            if self._exec_id != exec_id:
                return False
            self._text = text
            self._event.set()
            return True

    def cancel(self, exec_id: str | None = None, message: str = "Input cancelled by user") -> bool:
        with self._lock:
            if self._exec_id is None:
                return False
            if exec_id is not None and self._exec_id != exec_id:
                return False
            self._cancelled = True
            self._cancel_message = message
            self._event.set()
            return True

    def wait(self, exec_id: str, timeout: float = 300.0) -> str:
        with self._lock:
            event = self._event

        if not event.wait(timeout):
            self.cancel(exec_id, "Input timed out")

        with self._lock:
            try:
                if self._cancelled:
                    raise InputCancelledError(self._cancel_message)
                return self._text or ""
            finally:
                self._exec_id = None
                self._text = None
                self._cancelled = False
                self._cancel_message = "Input cancelled by user"
                self._event = threading.Event()

    def clear(self) -> None:
        with self._lock:
            self._exec_id = None
            self._text = None
            self._cancelled = False
            self._cancel_message = "Input cancelled by user"
            self._event = threading.Event()

    def current_exec_id(self) -> str | None:
        with self._lock:
            return self._exec_id


class KernelState:
    """Mutable kernel execution state shared across threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._executing = False
        self._request_id: int | None = None
        self._exec_id: str | None = None

    def begin_execution(self, request_id: int, exec_id: str | None) -> None:
        with self._lock:
            self._executing = True
            self._request_id = request_id
            self._exec_id = exec_id

    def finish_execution(self) -> None:
        with self._lock:
            self._executing = False
            self._request_id = None
            self._exec_id = None

    def snapshot(self) -> tuple[bool, int | None, str | None]:
        with self._lock:
            return self._executing, self._request_id, self._exec_id


class SnapshotStore:
    """Thread-safe store for the last committed runtime snapshot."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "variables": [],
            "nodes": {},
            "history": {"entries": [], "hasMore": False},
        }

    def replace(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._data = data

    def get(self) -> dict[str, Any]:
        with self._lock:
            return self._data


class KernelProcess:
    def __init__(self, cwd: str | None = None, assets_dir: str | None = None):
        self.writer = ProtocolWriter(os.dup(1))
        self.worker = IPythonWorker(cwd=cwd, assets_dir=assets_dir, venv=None)
        self.commands: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.inputs = InputMailbox()
        self.state = KernelState()
        self.snapshot = SnapshotStore()
        self._shutdown = threading.Event()

    # ------------------------------------------------------------------
    # Reader/control thread
    # ------------------------------------------------------------------

    def start_reader_thread(self) -> None:
        thread = threading.Thread(target=self._reader_loop, name="mrmd-kernel-reader", daemon=True)
        thread.start()

    def _reader_loop(self) -> None:
        busy_side_commands = {
            "complete",
            "inspect",
            "hover",
            "variables",
            "variable_detail",
            "history",
            "is_complete",
            "format",
        }

        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                self.writer.send({
                    "type": "protocol_error",
                    "error": {"type": "InvalidJSON", "message": str(exc)},
                })
                continue

            cmd = request.get("type", "")

            if cmd == "input_response":
                self.inputs.provide(request.get("execId", ""), request.get("text", ""))
                continue

            if cmd == "input_cancel":
                self.inputs.cancel(request.get("execId"), "Input cancelled by user")
                continue

            if cmd == "interrupt":
                self._handle_interrupt_command(request)
                continue

            if cmd == "shutdown":
                self._shutdown.set()
                self.inputs.cancel(message="Kernel shutting down")
                self.commands.put(None)
                return

            executing, _, _ = self.state.snapshot()
            if executing and cmd in busy_side_commands:
                self._handle_busy_side_request(request)
                continue

            self.commands.put(request)

        # Parent closed stdin.
        self._shutdown.set()
        self.inputs.cancel(message="Kernel control channel closed")
        self.commands.put(None)

    def _handle_busy_side_request(self, request: dict[str, Any]) -> None:
        request_id = request.get("id")
        try:
            result = self._dispatch_busy(request)
            if request_id is not None:
                self.writer.send({"type": "response", "id": request_id, "result": result})
        except BaseException as exc:
            if request_id is not None:
                self.writer.send({
                    "type": "response",
                    "id": request_id,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
                    },
                })

    def _handle_interrupt_command(self, request: dict[str, Any]) -> None:
        executing, _, exec_id = self.state.snapshot()
        waiting_exec_id = self.inputs.current_exec_id()
        interrupted = bool(executing or waiting_exec_id is not None)

        if waiting_exec_id is not None:
            self.inputs.cancel(waiting_exec_id, "Execution interrupted")

        # Best-effort same-process interrupt for non-POSIX environments.
        # On POSIX the parent also sends SIGINT directly to this process group,
        # which is what actually interrupts blocking calls like time.sleep().
        if interrupted and os.name == "nt":
            try:
                _thread.interrupt_main()
            except Exception:
                pass
            try:
                signal.raise_signal(signal.SIGINT)
            except Exception:
                pass

        request_id = request.get("id")
        if request_id is not None:
            self.writer.send({
                "type": "response",
                "id": request_id,
                "result": {"interrupted": interrupted, "execId": exec_id or waiting_exec_id},
            })

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _public_node(self, node: dict[str, Any], include_children: bool = False) -> dict[str, Any]:
        public = {
            "name": node.get("name"),
            "type": node.get("type"),
            "value": node.get("value"),
            "size": node.get("size"),
            "expandable": node.get("expandable", False),
            "shape": node.get("shape"),
            "dtype": node.get("dtype"),
            "length": node.get("length"),
            "keys": node.get("keys"),
        }
        if include_children:
            public.update({
                "fullValue": node.get("fullValue"),
                "children": [self._public_node(child) for child in (node.get("children") or [])] or None,
                "methods": node.get("methods"),
                "attributes": node.get("attributes"),
            })
        return public

    def _infer_kind(self, value: Any) -> str:
        try:
            if inspect.ismodule(value):
                return "module"
            if inspect.isclass(value):
                return "class"
            if inspect.ismethod(value):
                return "method"
            if inspect.isfunction(value) or inspect.isbuiltin(value):
                return "function"
            if isinstance(value, property):
                return "property"
        except Exception:
            pass
        return "variable"

    def _safe_signature(self, value: Any, kind: str) -> str | None:
        if kind not in {"function", "method", "class"}:
            return None
        try:
            return str(inspect.signature(value))
        except Exception:
            return None

    def _safe_docstring(self, value: Any, kind: str) -> str | None:
        if kind not in {"function", "method", "class", "module"}:
            return None
        try:
            return inspect.getdoc(value)
        except Exception:
            return None

    def _safe_source_location(self, value: Any, kind: str) -> tuple[str | None, int | None]:
        if kind not in {"function", "method", "class", "module"}:
            return None, None
        try:
            file = inspect.getsourcefile(value) or inspect.getfile(value)
        except Exception:
            file = None
        try:
            _, line = inspect.getsourcelines(value)
        except Exception:
            line = None
        return file, line

    def _snapshot_node(self, display_name: str, token: str, value: Any, depth: int, seen: set[int]) -> dict[str, Any]:
        var = self.worker._make_variable(display_name, value)
        kind = self._infer_kind(value)
        file, line = self._safe_source_location(value, kind)
        node: dict[str, Any] = {
            "name": var.name,
            "token": token,
            "kind": kind,
            "type": var.type,
            "value": var.value,
            "fullValue": var.value,
            "size": var.size,
            "expandable": var.expandable,
            "shape": var.shape,
            "dtype": var.dtype,
            "length": var.length,
            "keys": var.keys,
            "signature": self._safe_signature(value, kind),
            "docstring": self._safe_docstring(value, kind),
            "file": file,
            "line": line,
            "children": None,
            "methods": None,
            "attributes": None,
            "_childrenByToken": {},
        }

        if depth >= SNAPSHOT_MAX_DEPTH:
            return node

        obj_id = id(value)
        if obj_id in seen:
            return node

        children: list[dict[str, Any]] = []
        children_by_token: dict[str, dict[str, Any]] = {}
        methods: list[str] = []
        attributes: list[str] = []

        seen.add(obj_id)
        try:
            if isinstance(value, dict):
                items = list(value.items())[:SNAPSHOT_MAX_CHILDREN]
                for raw_key, child_value in items:
                    token_key = str(raw_key)
                    child_display_name = repr(raw_key)
                    child = self._snapshot_node(child_display_name, token_key, child_value, depth + 1, seen)
                    children.append(child)
                    children_by_token[token_key] = child
                    children_by_token[child_display_name] = child
            elif isinstance(value, (list, tuple)):
                items = list(value)[:SNAPSHOT_MAX_CHILDREN]
                for index, child_value in enumerate(items):
                    token_key = str(index)
                    child_display_name = f"[{index}]"
                    child = self._snapshot_node(child_display_name, token_key, child_value, depth + 1, seen)
                    children.append(child)
                    children_by_token[token_key] = child
                    children_by_token[child_display_name] = child
            else:
                member_values: dict[str, Any] = {}
                try:
                    for key, child_value in vars(value).items():
                        if key.startswith("_"):
                            continue
                        member_values[key] = child_value
                        attributes.append(key)
                except Exception:
                    pass

                class_dict = getattr(type(value), "__dict__", {})
                for key, member in class_dict.items():
                    if key.startswith("_") or key in member_values:
                        continue
                    raw_member = member
                    if isinstance(member, (staticmethod, classmethod)):
                        raw_member = member.__func__
                    member_values[key] = raw_member
                    if callable(raw_member):
                        methods.append(key)
                    else:
                        attributes.append(key)

                for key, child_value in list(member_values.items())[:SNAPSHOT_MAX_CHILDREN]:
                    child = self._snapshot_node(key, key, child_value, depth + 1, seen)
                    children.append(child)
                    children_by_token[key] = child

        finally:
            seen.discard(obj_id)

        if children:
            node["children"] = children
            node["_childrenByToken"] = children_by_token
        if methods:
            node["methods"] = sorted(set(methods))[:SNAPSHOT_MAX_CHILDREN]
        if attributes:
            node["attributes"] = sorted(set(attributes))[:SNAPSHOT_MAX_CHILDREN]

        return node

    def _refresh_snapshot(self) -> None:
        """Capture a fresh committed snapshot of namespace + history.

        Called on the kernel main thread after executions and resets.
        """
        try:
            self.worker._ensure_initialized()
            variables_result = self.worker.get_variables()
            user_ns = self.worker.shell.user_ns if self.worker.shell is not None else {}

            nodes: dict[str, dict[str, Any]] = {}
            variables: list[dict[str, Any]] = []

            for variable in variables_result.variables:
                if variable.name not in user_ns:
                    continue
                node = self._snapshot_node(variable.name, variable.name, user_ns[variable.name], 0, set())
                nodes[variable.name] = node
                variables.append(self._public_node(node))

            history = _to_jsonable(self.worker.get_history(n=200, pattern=None, before=None))
            self.snapshot.replace({
                "variables": variables,
                "nodes": nodes,
                "history": history,
            })
        except Exception:
            # Keep the last good snapshot if refresh fails.
            pass

    def _resolve_snapshot_node(self, path_expr: str | None) -> dict[str, Any] | None:
        if not path_expr:
            return None

        snapshot = self.snapshot.get()
        current = snapshot.get("nodes", {}).get(path_expr.split(".", 1)[0])
        if current is None:
            return None

        parts = path_expr.split(".")
        for token in parts[1:]:
            child_map = current.get("_childrenByToken", {})
            current = child_map.get(token) or child_map.get(token.strip("[]"))
            if current is None:
                return None
        return current

    def _snapshot_complete(self, code: str, cursor_pos: int) -> dict[str, Any]:
        name = self.worker._extract_name_at_cursor(code, cursor_pos) or ""
        snapshot = self.snapshot.get()
        matches: list[dict[str, Any]] = []

        if "." in name:
            base_expr, prefix = name.rsplit(".", 1)
            base_node = self._resolve_snapshot_node(base_expr)
            if base_node is not None:
                labels: set[str] = set()
                labels.update(base_node.get("_childrenByToken", {}).keys())
                for label in base_node.get("methods") or []:
                    labels.add(label)
                for label in base_node.get("attributes") or []:
                    labels.add(label)

                filtered_labels = [
                    label for label in sorted(labels)
                    if label.startswith(prefix) and not label.startswith("[")
                ]
                for label in filtered_labels:
                    child = base_node.get("_childrenByToken", {}).get(label)
                    matches.append({
                        "label": label,
                        "kind": child.get("kind", "variable") if child else "variable",
                        "type": child.get("type") if child else None,
                        "detail": child.get("type") if child else None,
                        "documentation": child.get("docstring") if child else None,
                        "valuePreview": child.get("value") if child else None,
                    })
                return {
                    "matches": matches,
                    "cursorStart": cursor_pos - len(prefix),
                    "cursorEnd": cursor_pos,
                    "source": "runtime",
                }

        prefix = name
        for label, node in sorted(snapshot.get("nodes", {}).items()):
            if not label.startswith(prefix):
                continue
            matches.append({
                "label": label,
                "kind": node.get("kind", "variable"),
                "type": node.get("type"),
                "detail": node.get("type"),
                "documentation": node.get("docstring"),
                "valuePreview": node.get("value"),
            })

        return {
            "matches": matches,
            "cursorStart": cursor_pos - len(prefix),
            "cursorEnd": cursor_pos,
            "source": "runtime",
        }

    def _snapshot_hover(self, code: str, cursor_pos: int) -> dict[str, Any]:
        name = self.worker._extract_name_at_cursor(code, cursor_pos)
        node = self._resolve_snapshot_node(name)
        if node is None:
            return {"found": False}
        return {
            "found": True,
            "name": name,
            "type": node.get("type"),
            "value": node.get("value"),
            "signature": node.get("signature"),
            "docstring": node.get("docstring"),
        }

    def _snapshot_inspect(self, code: str, cursor_pos: int, detail: int = 1) -> dict[str, Any]:
        name = self.worker._extract_name_at_cursor(code, cursor_pos)
        node = self._resolve_snapshot_node(name)
        if node is None:
            return {"found": False, "name": name}
        return {
            "found": True,
            "source": "runtime",
            "name": name,
            "kind": node.get("kind"),
            "type": node.get("type"),
            "signature": node.get("signature"),
            "docstring": node.get("docstring") if detail >= 1 else None,
            "sourceCode": None,
            "file": node.get("file") if detail >= 2 else None,
            "line": node.get("line") if detail >= 2 else None,
            "value": node.get("value"),
            "children": len(node.get("children") or []),
        }

    def _snapshot_variables(self) -> dict[str, Any]:
        snapshot = self.snapshot.get()
        variables = list(snapshot.get("variables", []))
        return {
            "variables": variables,
            "count": len(variables),
            "truncated": False,
            "warnings": [STALENESS_WARNING],
        }

    def _snapshot_variable_detail(self, name: str, path: list[str] | None = None) -> dict[str, Any]:
        node = self._resolve_snapshot_node(name)
        if node is None:
            return {
                "name": name,
                "type": "unknown",
                "value": "<not found>",
                "expandable": False,
                "children": None,
            }

        for token in path or []:
            child_map = node.get("_childrenByToken", {})
            normalized = token.strip("[]")
            node = child_map.get(token) or child_map.get(normalized)
            if node is None:
                return {
                    "name": normalized,
                    "type": "unknown",
                    "value": "<not found>",
                    "expandable": False,
                    "children": None,
                }

        return self._public_node(node, include_children=True)

    def _snapshot_history(self, n: int = 20, before: int | None = None) -> dict[str, Any]:
        snapshot = self.snapshot.get().get("history", {"entries": [], "hasMore": False})
        entries = list(snapshot.get("entries", []))
        if before is not None:
            entries = [entry for entry in entries if entry.get("historyIndex", 0) < before]
        if n <= 0:
            n = 20
        start = max(0, len(entries) - n)
        return {
            "entries": entries[start:],
            "hasMore": start > 0,
        }

    def _busy_is_complete(self, code: str) -> dict[str, Any]:
        try:
            from IPython.core.inputtransformer2 import TransformerManager
            status, indent = TransformerManager().check_complete(code)
            return {"status": status, "indent": indent or ""}
        except Exception:
            return {"status": "unknown", "indent": ""}

    def _busy_format(self, code: str) -> dict[str, Any]:
        try:
            import black
            formatted = black.format_str(code, mode=black.Mode())
            return {"formatted": formatted, "changed": formatted != code}
        except ImportError:
            return {"formatted": code, "changed": False}
        except Exception:
            return {"formatted": code, "changed": False}

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _stdin_callback(self, request_id: int, exec_id: str | None, stdin_request: StdinRequest) -> str:
        effective_exec_id = stdin_request.execId or exec_id or ""
        self.inputs.begin(effective_exec_id)
        self.writer.send({
            "type": "event",
            "id": request_id,
            "event": "stdin_request",
            "data": _to_jsonable(stdin_request),
        })
        return self.inputs.wait(effective_exec_id)

    def _handle_execute(self, request: dict[str, Any]) -> dict[str, Any]:
        return _to_jsonable(self.worker.execute(
            code=request.get("code", ""),
            store_history=request.get("storeHistory", True),
            exec_id=request.get("execId"),
            allow_stdin=request.get("allowStdin", False),
        ))

    def _handle_execute_stream(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = int(request["id"])
        exec_id = request.get("execId")

        def on_output(stream: str, content: str, _accumulated: str) -> None:
            self.writer.send({
                "type": "event",
                "id": request_id,
                "event": stream,
                "data": {"content": content},
            })

        def on_stdin_request(stdin_request: StdinRequest) -> str:
            return self._stdin_callback(request_id, exec_id, stdin_request)

        return _to_jsonable(self.worker.execute_streaming(
            code=request.get("code", ""),
            on_output=on_output,
            store_history=request.get("storeHistory", True),
            exec_id=exec_id,
            on_stdin_request=on_stdin_request if request.get("allowStdin", True) else None,
            allow_stdin=request.get("allowStdin", True),
        ))

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        cmd = request.get("type", "")

        if cmd == "execute":
            return self._handle_execute(request)
        if cmd == "execute_stream":
            return self._handle_execute_stream(request)
        if cmd == "complete":
            return _to_jsonable(self.worker.complete(request.get("code", ""), request.get("cursor", 0)))
        if cmd == "inspect":
            return _to_jsonable(self.worker.inspect(
                request.get("code", ""),
                request.get("cursor", 0),
                request.get("detail", 1),
            ))
        if cmd == "hover":
            return _to_jsonable(self.worker.hover(request.get("code", ""), request.get("cursor", 0)))
        if cmd == "variables":
            return _to_jsonable(self.worker.get_variables())
        if cmd == "variable_detail":
            return _to_jsonable(self.worker.get_variable_detail(
                request.get("name", ""),
                request.get("path", []),
            ))
        if cmd == "reset":
            self.worker.reset()
            self._refresh_snapshot()
            return {"success": True}
        if cmd == "clear_history":
            success = bool(self.worker.clear_history())
            self._refresh_snapshot()
            return {"success": success}
        if cmd == "is_complete":
            return _to_jsonable(self.worker.is_complete(request.get("code", "")))
        if cmd == "format":
            formatted, changed = self.worker.format_code(request.get("code", ""))
            return {"formatted": formatted, "changed": changed}
        if cmd == "history":
            return _to_jsonable(self.worker.get_history(
                n=request.get("n", 20),
                pattern=request.get("pattern"),
                before=request.get("before"),
            ))
        if cmd == "info":
            return self.worker.get_info()

        raise ValueError(f"Unknown command: {cmd}")

    def _dispatch_busy(self, request: dict[str, Any]) -> dict[str, Any]:
        cmd = request.get("type", "")
        if cmd == "complete":
            return self._snapshot_complete(request.get("code", ""), request.get("cursor", 0))
        if cmd == "inspect":
            return self._snapshot_inspect(
                request.get("code", ""),
                request.get("cursor", 0),
                request.get("detail", 1),
            )
        if cmd == "hover":
            return self._snapshot_hover(request.get("code", ""), request.get("cursor", 0))
        if cmd == "variables":
            return self._snapshot_variables()
        if cmd == "variable_detail":
            return self._snapshot_variable_detail(request.get("name", ""), request.get("path", []))
        if cmd == "history":
            return self._snapshot_history(
                n=request.get("n", 20),
                before=request.get("before"),
            )
        if cmd == "is_complete":
            return self._busy_is_complete(request.get("code", ""))
        if cmd == "format":
            return self._busy_format(request.get("code", ""))
        raise ValueError(f"Unsupported busy-side command: {cmd}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def serve_forever(self) -> int:
        self.start_reader_thread()
        self.writer.send({"type": "ready", "pid": os.getpid()})

        while not self._shutdown.is_set():
            try:
                request = self.commands.get()
            except KeyboardInterrupt:
                # Late-arriving interrupt after an execution already finished.
                continue

            if request is None:
                break

            request_id = request.get("id")
            cmd = request.get("type", "")
            is_execution = cmd in {"execute", "execute_stream"}
            exec_id = request.get("execId")

            if is_execution and request_id is not None:
                self.state.begin_execution(int(request_id), exec_id)

            try:
                result = self._dispatch(request)
                if request_id is not None:
                    self.writer.send({"type": "response", "id": request_id, "result": result})
            except KeyboardInterrupt:
                if request_id is not None:
                    result = ExecuteResult(
                        success=False,
                        error=ExecuteError(type="KeyboardInterrupt", message="Interrupted"),
                    )
                    self.writer.send({
                        "type": "response",
                        "id": request_id,
                        "result": _to_jsonable(result),
                    })
            except BaseException as exc:
                if request_id is not None:
                    self.writer.send({
                        "type": "response",
                        "id": request_id,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
                        },
                    })
            finally:
                if is_execution:
                    self.inputs.clear()
                    self._refresh_snapshot()
                    self.state.finish_execution()

        return 0


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="rat-py kernel process")
    parser.add_argument("--cwd", default=None, help="Working directory")
    parser.add_argument("--assets-dir", default=None, help="Assets directory")
    args = parser.parse_args()

    kernel = KernelProcess(cwd=args.cwd, assets_dir=args.assets_dir)
    return kernel.serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
