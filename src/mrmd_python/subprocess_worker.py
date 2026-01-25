"""
Subprocess Worker - Runs inside the venv subprocess.

This module is executed as a persistent subprocess using the venv's Python.
It communicates with the parent process via JSON over stdin/stdout.

The subprocess holds an IPython shell, so variables persist across executions.
Killing this subprocess releases all resources (including GPU memory).

Protocol:
- Parent sends JSON commands to stdin (one per line)
- Worker sends JSON responses to stdout (one per line)
- Stderr is reserved for streaming output during execution

Usage:
    /path/to/venv/bin/python -m mrmd_python.subprocess_worker --cwd /project --assets-dir /assets
"""

import sys
import os
import json
import traceback
import time
import io
import base64
import threading
import select
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Asset:
    path: str
    url: str
    mimeType: str
    assetType: str
    size: int


@dataclass
class ExecuteError:
    type: str
    message: str
    traceback: list[str] = field(default_factory=list)
    line: int | None = None
    column: int | None = None


@dataclass
class ExecuteResult:
    success: bool = True
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: ExecuteError | None = None
    displayData: list[dict] = field(default_factory=list)
    assets: list[Asset] = field(default_factory=list)
    executionCount: int = 0
    duration: int | None = None


class SubprocessIPythonWorker:
    """
    IPython worker running inside a subprocess.

    This is similar to IPythonWorker but designed to run as the main
    process in a venv subprocess, communicating via stdin/stdout.
    """

    def __init__(self, cwd: str | None = None, assets_dir: str | None = None):
        self.cwd = cwd
        self.assets_dir = assets_dir
        self.shell = None
        self._initialized = False
        self._captured_displays: list[dict] = []
        self._asset_counter = 0
        self._current_exec_id: str | None = None
        self._execution_count = 0

        # For streaming output
        self._output_callback = None

        # Set matplotlib backend to Agg early (before any imports)
        # This ensures plots work even if matplotlib is installed later via %pip
        os.environ.setdefault("MPLBACKEND", "Agg")

    def _ensure_initialized(self):
        """Lazy initialization of IPython shell."""
        if self._initialized:
            return

        from IPython.core.interactiveshell import InteractiveShell

        self.shell = InteractiveShell.instance()

        # Enable rich display
        self.shell.display_formatter.active_types = [
            "text/plain",
            "text/html",
            "text/markdown",
            "text/latex",
            "image/png",
            "image/jpeg",
            "image/svg+xml",
            "application/json",
            "application/javascript",
        ]

        # Set up display capture
        self._setup_display_capture()

        # Set up matplotlib hook
        self._setup_matplotlib_hook()

        # Set up Plotly
        self._setup_plotly_renderer()

        # Change to working directory
        if self.cwd:
            os.chdir(self.cwd)
            if self.cwd not in sys.path:
                sys.path.insert(0, self.cwd)
            src_dir = os.path.join(self.cwd, "src")
            if os.path.isdir(src_dir) and src_dir not in sys.path:
                sys.path.insert(0, src_dir)

        # Enable autoreload
        self._setup_autoreload()

        self._initialized = True

    def _setup_plotly_renderer(self):
        """Configure Plotly to render as HTML."""
        try:
            import plotly.io as pio
            pio.renderers.default = "notebook_connected"
        except ImportError:
            pass

    def _setup_autoreload(self):
        """Enable autoreload extension."""
        try:
            self.shell.run_line_magic("load_ext", "autoreload")
            self.shell.run_line_magic("autoreload", "2")
        except Exception:
            pass

    def _save_asset(self, content: bytes, mime_type: str, extension: str) -> Asset | None:
        """Save content as an asset file."""
        if not self.assets_dir:
            return None

        assets_path = Path(self.assets_dir)
        assets_path.mkdir(parents=True, exist_ok=True)

        self._asset_counter += 1
        if self._current_exec_id:
            filename = f"{self._current_exec_id}_{self._asset_counter:04d}.{extension}"
        else:
            filename = f"output_{self._asset_counter:04d}.{extension}"

        filepath = assets_path / filename
        filepath.write_bytes(content)

        asset_type = "file"
        if mime_type.startswith("image/"):
            asset_type = "image"
        elif mime_type == "text/html":
            asset_type = "html"
        elif mime_type == "image/svg+xml":
            asset_type = "svg"

        return Asset(
            path=str(filepath),
            url=f"/mrp/v1/assets/{filename}",
            mimeType=mime_type,
            assetType=asset_type,
            size=len(content),
        )

    def _setup_display_capture(self):
        """Set up capture of rich display outputs."""
        worker = self

        class DisplayCapture:
            def __init__(self, shell):
                self.shell = shell
                self.is_publishing = False

            def publish(self, data, metadata=None, source=None, **kwargs):
                self._process_display(data, metadata or {})

            def _process_display(self, data, metadata):
                # Handle image/png
                if "image/png" in data:
                    png_data = data["image/png"]
                    if isinstance(png_data, str):
                        png_bytes = base64.b64decode(png_data)
                    else:
                        png_bytes = png_data

                    asset = worker._save_asset(png_bytes, "image/png", "png")
                    if asset:
                        worker._captured_displays.append({"asset": asdict(asset)})
                    else:
                        worker._captured_displays.append({"data": data, "metadata": metadata})
                    return

                # Handle text/html
                if "text/html" in data:
                    html_content = data["text/html"]
                    if html_content and html_content.strip():
                        worker._captured_displays.append({"data": data, "metadata": metadata})
                    return

                # Handle text/plain
                if "text/plain" in data:
                    worker._captured_displays.append(
                        {"data": {"text/plain": data["text/plain"]}, "metadata": metadata}
                    )
                    return

                if data:
                    worker._captured_displays.append({"data": data, "metadata": metadata})

            def clear_output(self, wait=False):
                pass

        self.shell.display_pub = DisplayCapture(self.shell)

    def _setup_matplotlib_hook(self):
        """Set up matplotlib to save figures as assets."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            worker = self

            def _hooked_show(*args, **kwargs):
                if worker.assets_dir:
                    from io import BytesIO

                    assets_path = Path(worker.assets_dir)
                    assets_path.mkdir(parents=True, exist_ok=True)

                    for num in plt.get_fignums():
                        fig = plt.figure(num)
                        buf = BytesIO()
                        fig.savefig(
                            buf,
                            format="png",
                            dpi=150,
                            bbox_inches="tight",
                            facecolor="white",
                            edgecolor="none",
                        )
                        buf.seek(0)
                        png_bytes = buf.getvalue()

                        worker._asset_counter += 1
                        if worker._current_exec_id:
                            filename = f"{worker._current_exec_id}_{worker._asset_counter:04d}.png"
                        else:
                            filename = f"figure_{worker._asset_counter:04d}.png"
                        filepath = assets_path / filename
                        filepath.write_bytes(png_bytes)

                        asset = Asset(
                            path=str(filepath),
                            url=f"/mrp/v1/assets/{filename}",
                            mimeType="image/png",
                            assetType="image",
                            size=len(png_bytes),
                        )
                        worker._captured_displays.append({"asset": asdict(asset)})

                plt.close("all")

            _hooked_show._mrmd_hooked = True
            plt.show = _hooked_show
            if "matplotlib.pyplot" in sys.modules:
                sys.modules["matplotlib.pyplot"].show = _hooked_show

        except ImportError:
            pass

    def _ensure_matplotlib_hook(self):
        """Re-apply matplotlib hook after user imports it."""
        if "matplotlib.pyplot" not in sys.modules:
            return

        plt = sys.modules["matplotlib.pyplot"]
        if hasattr(plt.show, "_mrmd_hooked"):
            return

        worker = self

        def _hooked_show(*args, **kwargs):
            if worker.assets_dir:
                from io import BytesIO

                assets_path = Path(worker.assets_dir)
                assets_path.mkdir(parents=True, exist_ok=True)

                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    buf = BytesIO()
                    fig.savefig(
                        buf,
                        format="png",
                        dpi=150,
                        bbox_inches="tight",
                        facecolor="white",
                        edgecolor="none",
                    )
                    buf.seek(0)
                    png_bytes = buf.getvalue()

                    worker._asset_counter += 1
                    if worker._current_exec_id:
                        filename = (
                            f"{worker._current_exec_id}_{worker._asset_counter:04d}.png"
                        )
                    else:
                        filename = f"figure_{worker._asset_counter:04d}.png"
                    filepath = assets_path / filename
                    filepath.write_bytes(png_bytes)

                    asset = Asset(
                        path=str(filepath),
                        url=f"/mrp/v1/assets/{filename}",
                        mimeType="image/png",
                        assetType="image",
                        size=len(png_bytes),
                    )
                    worker._captured_displays.append({"asset": asdict(asset)})

            plt.close("all")

        _hooked_show._mrmd_hooked = True
        plt.show = _hooked_show

    def execute(self, code: str, store_history: bool = True, exec_id: str | None = None) -> ExecuteResult:
        """Execute code and return result."""
        self._ensure_initialized()
        self._captured_displays = []
        self._current_exec_id = exec_id
        self._asset_counter = 0

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = captured_stdout = io.StringIO()
        sys.stderr = captured_stderr = io.StringIO()

        start_time = time.time()
        result = ExecuteResult()

        try:
            # Ensure matplotlib hook is applied BEFORE execution
            # (handles case where matplotlib was installed via %pip)
            self._ensure_matplotlib_hook()

            exec_result = self.shell.run_cell(code, store_history=store_history, silent=False)

            # Also check after, in case user imported matplotlib during this cell
            self._ensure_matplotlib_hook()

            result.executionCount = self.shell.execution_count
            result.success = exec_result.success

            if exec_result.result is not None:
                obj = exec_result.result
                try:
                    if hasattr(obj, "_repr_html_"):
                        from IPython.display import display
                        display(obj)
                    elif hasattr(obj, "_repr_png_"):
                        from IPython.display import display
                        display(obj)
                    result.result = repr(obj)
                except Exception:
                    result.result = "<repr failed>"

            if exec_result.error_in_exec:
                result.error = self._format_exception(exec_result.error_in_exec)
                result.success = False
            elif exec_result.error_before_exec:
                result.error = self._format_exception(exec_result.error_before_exec)
                result.success = False

            # Convert display data
            for disp in self._captured_displays:
                if "asset" in disp:
                    result.assets.append(Asset(**disp["asset"]))
                elif "data" in disp:
                    result.displayData.append(disp)

        except Exception as e:
            result.error = self._format_exception(e)
            result.success = False

        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            result.stdout = captured_stdout.getvalue()
            result.stderr = captured_stderr.getvalue()
            result.duration = int((time.time() - start_time) * 1000)
            self._current_exec_id = None

        return result

    def execute_streaming(
        self,
        code: str,
        output_fd: int,
        store_history: bool = True,
        exec_id: str | None = None,
    ) -> ExecuteResult:
        """Execute code with streaming output to the given file descriptor."""
        self._ensure_initialized()
        self._captured_displays = []
        self._current_exec_id = exec_id
        self._asset_counter = 0

        # Use PTY for proper terminal emulation if available
        try:
            import pty
            stdout_master, stdout_slave = pty.openpty()
            stderr_master, stderr_slave = pty.openpty()
            use_pty = True
        except (ImportError, OSError):
            stdout_master, stdout_slave = os.pipe()
            stderr_master, stderr_slave = os.pipe()
            use_pty = False

        # Save original FDs
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)

        # Redirect stdout/stderr to our pipes
        os.dup2(stdout_slave, 1)
        os.dup2(stderr_slave, 2)
        os.close(stdout_slave)
        os.close(stderr_slave)

        # Make master FDs non-blocking
        try:
            import fcntl
            for fd in [stdout_master, stderr_master]:
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except ImportError:
            pass

        accumulated_stdout = []
        accumulated_stderr = []
        stop_reader = threading.Event()
        start_time = time.time()

        def reader_thread():
            """Read output and send to parent via output_fd."""
            output_file = os.fdopen(output_fd, 'w', buffering=1)

            while not stop_reader.is_set():
                try:
                    readable, _, _ = select.select([stdout_master, stderr_master], [], [], 0.05)
                except (ValueError, OSError):
                    break

                for fd in readable:
                    try:
                        data = os.read(fd, 4096)
                        if not data:
                            continue

                        stream = "stdout" if fd == stdout_master else "stderr"
                        text = data.decode("utf-8", errors="replace")

                        if fd == stdout_master:
                            accumulated_stdout.append(text)
                        else:
                            accumulated_stderr.append(text)

                        # Send streaming event to parent
                        event = {
                            "type": "stream",
                            "stream": stream,
                            "content": text,
                            "accumulated": "".join(accumulated_stdout if fd == stdout_master else accumulated_stderr),
                        }
                        output_file.write(json.dumps(event) + "\n")
                        output_file.flush()

                    except (BlockingIOError, OSError):
                        pass

            # Final read
            for fd, stream, acc in [
                (stdout_master, "stdout", accumulated_stdout),
                (stderr_master, "stderr", accumulated_stderr),
            ]:
                try:
                    while True:
                        data = os.read(fd, 4096)
                        if not data:
                            break
                        text = data.decode("utf-8", errors="replace")
                        acc.append(text)
                        event = {
                            "type": "stream",
                            "stream": stream,
                            "content": text,
                            "accumulated": "".join(acc),
                        }
                        output_file.write(json.dumps(event) + "\n")
                        output_file.flush()
                except (BlockingIOError, OSError):
                    pass

            output_file.close()

        reader = threading.Thread(target=reader_thread, daemon=True)
        reader.start()

        # Update sys.stdout/stderr
        sys.stdout = io.TextIOWrapper(io.FileIO(1, mode="w", closefd=False), line_buffering=True)
        sys.stderr = io.TextIOWrapper(io.FileIO(2, mode="w", closefd=False), line_buffering=True)

        result = ExecuteResult()

        try:
            # Ensure matplotlib hook is applied BEFORE execution
            # (handles case where matplotlib was installed via %pip)
            self._ensure_matplotlib_hook()

            exec_result = self.shell.run_cell(code, store_history=store_history, silent=False)

            # Also check after, in case user imported matplotlib during this cell
            self._ensure_matplotlib_hook()

            result.executionCount = self.shell.execution_count
            result.success = exec_result.success

            if exec_result.result is not None:
                obj = exec_result.result
                try:
                    if hasattr(obj, "_repr_html_"):
                        from IPython.display import display
                        display(obj)
                    elif hasattr(obj, "_repr_png_"):
                        from IPython.display import display
                        display(obj)
                    result.result = repr(obj)
                except Exception:
                    result.result = "<repr failed>"

            if exec_result.error_in_exec:
                result.error = self._format_exception(exec_result.error_in_exec)
                result.success = False
            elif exec_result.error_before_exec:
                result.error = self._format_exception(exec_result.error_before_exec)
                result.success = False

            for disp in self._captured_displays:
                if "asset" in disp:
                    result.assets.append(Asset(**disp["asset"]))
                elif "data" in disp:
                    result.displayData.append(disp)

        except KeyboardInterrupt:
            result.error = ExecuteError(type="KeyboardInterrupt", message="Interrupted")
            result.success = False
        except Exception as e:
            result.error = self._format_exception(e)
            result.success = False

        finally:
            sys.stdout.flush()
            sys.stderr.flush()

            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)

            stop_reader.set()
            reader.join(timeout=1.0)

            os.close(stdout_master)
            os.close(stderr_master)

            sys.stdout = io.TextIOWrapper(io.FileIO(1, mode="w", closefd=False), line_buffering=True)
            sys.stderr = io.TextIOWrapper(io.FileIO(2, mode="w", closefd=False), line_buffering=True)

            result.stdout = "".join(accumulated_stdout)
            result.stderr = "".join(accumulated_stderr)
            result.duration = int((time.time() - start_time) * 1000)
            self._current_exec_id = None

        return result

    def complete(self, code: str, cursor_pos: int) -> dict:
        """Get completions at cursor position."""
        self._ensure_initialized()

        try:
            from IPython.core.completer import provisionalcompleter

            with provisionalcompleter():
                completions = list(self.shell.Completer.completions(code, cursor_pos))

            if completions:
                items = []
                for c in completions:
                    items.append({
                        "label": c.text,
                        "kind": c.type or "variable",
                        "type": c.type,
                    })
                return {
                    "matches": items,
                    "cursorStart": completions[0].start,
                    "cursorEnd": completions[0].end,
                    "source": "runtime",
                }
        except Exception:
            pass

        return {"matches": [], "cursorStart": cursor_pos, "cursorEnd": cursor_pos, "source": "runtime"}

    def inspect(self, code: str, cursor_pos: int, detail: int = 1) -> dict:
        """Get detailed info about symbol at cursor."""
        self._ensure_initialized()

        try:
            name = self._extract_name_at_cursor(code, cursor_pos)
            if not name:
                return {"found": False}

            info = self.shell.object_inspect(name)
            if not info.get("found"):
                return {"found": False, "name": name}

            return {
                "found": True,
                "source": "runtime",
                "name": name,
                "type": info.get("type_name"),
                "signature": info.get("call_signature") or info.get("init_signature"),
                "docstring": info.get("docstring") if detail >= 1 else None,
                "sourceCode": info.get("source") if detail >= 2 else None,
                "file": info.get("file"),
                "line": info.get("line"),
            }
        except Exception:
            return {"found": False}

    def hover(self, code: str, cursor_pos: int) -> dict:
        """Get hover tooltip for symbol."""
        self._ensure_initialized()

        try:
            name = self._extract_name_at_cursor(code, cursor_pos)
            if not name:
                return {"found": False}

            try:
                value = eval(name, {"__builtins__": {}}, self.shell.user_ns)
                type_name = type(value).__name__

                # Get value preview
                preview = repr(value)
                if len(preview) > 200:
                    preview = preview[:197] + "..."

                return {
                    "found": True,
                    "name": name,
                    "type": type_name,
                    "value": preview,
                }
            except Exception:
                pass

            info = self.shell.object_inspect(name)
            if info.get("found"):
                return {
                    "found": True,
                    "name": name,
                    "type": info.get("type_name"),
                    "signature": info.get("call_signature"),
                }

            return {"found": False}
        except Exception:
            return {"found": False}

    def get_variables(self) -> dict:
        """Get user variables."""
        self._ensure_initialized()

        variables = []
        user_ns = self.shell.user_ns

        skip_prefixes = ("_", "In", "Out", "get_ipython", "exit", "quit")

        import builtins
        builtin_names = set(dir(builtins))

        for name, value in user_ns.items():
            if name.startswith(skip_prefixes):
                continue
            if name in builtin_names:
                continue

            try:
                type_name = type(value).__name__
                preview = repr(value)
                if len(preview) > 80:
                    preview = preview[:77] + "..."

                var = {
                    "name": name,
                    "type": type_name,
                    "value": preview,
                    "expandable": isinstance(value, (dict, list, tuple)) or hasattr(value, "__dict__"),
                }

                if hasattr(value, "shape"):
                    var["shape"] = list(value.shape)
                if hasattr(value, "__len__"):
                    try:
                        var["length"] = len(value)
                    except Exception:
                        pass

                variables.append(var)
            except Exception:
                pass

        variables.sort(key=lambda v: v["name"])
        return {"variables": variables, "count": len(variables), "truncated": False}

    def reset(self):
        """Reset the namespace."""
        self._ensure_initialized()
        self.shell.reset()

    def is_complete(self, code: str) -> dict:
        """Check if code is a complete statement."""
        self._ensure_initialized()

        try:
            status, indent = self.shell.input_transformer_manager.check_complete(code)
            return {"status": status, "indent": indent or ""}
        except Exception:
            return {"status": "unknown", "indent": ""}

    def _extract_name_at_cursor(self, code: str, cursor_pos: int) -> str | None:
        """Extract Python name at cursor."""
        if cursor_pos > len(code):
            cursor_pos = len(code)

        start = cursor_pos
        while start > 0 and (code[start - 1].isalnum() or code[start - 1] in "_."):
            start -= 1

        end = cursor_pos
        while end < len(code) and (code[end].isalnum() or code[end] == "_"):
            end += 1

        name = code[start:end]
        if name and (name[0].isalpha() or name[0] == "_"):
            return name
        return None

    def _format_exception(self, exc: Exception) -> ExecuteError:
        """Format exception to ExecuteError."""
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        return ExecuteError(
            type=type(exc).__name__,
            message=str(exc),
            traceback=tb_lines,
        )


def _dataclass_to_dict(obj: Any) -> dict:
    """Convert dataclass to dict recursively."""
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


def main():
    """Main entry point for subprocess worker."""
    import argparse

    parser = argparse.ArgumentParser(description="MRMD Python Subprocess Worker")
    parser.add_argument("--cwd", default=None, help="Working directory")
    parser.add_argument("--assets-dir", default=None, help="Assets directory")
    args = parser.parse_args()

    worker = SubprocessIPythonWorker(cwd=args.cwd, assets_dir=args.assets_dir)

    # Signal ready
    sys.stdout.write(json.dumps({"type": "ready", "pid": os.getpid()}) + "\n")
    sys.stdout.flush()

    # Main command loop
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = {"type": "error", "error": f"Invalid JSON: {e}"}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue

        cmd = request.get("type", "")

        try:
            if cmd == "execute":
                result = worker.execute(
                    code=request.get("code", ""),
                    store_history=request.get("storeHistory", True),
                    exec_id=request.get("execId"),
                )
                response = {"type": "result", "result": _dataclass_to_dict(result)}

            elif cmd == "execute_stream":
                # For streaming, we use a pipe to send output events
                # The parent will read from the pipe while we execute
                read_fd, write_fd = os.pipe()

                # Tell parent about the pipe FD (they'll read from their end)
                sys.stdout.write(json.dumps({"type": "stream_start", "pipe_fd": read_fd}) + "\n")
                sys.stdout.flush()

                # Execute with streaming to the write end
                result = worker.execute_streaming(
                    code=request.get("code", ""),
                    output_fd=write_fd,
                    store_history=request.get("storeHistory", True),
                    exec_id=request.get("execId"),
                )

                response = {"type": "result", "result": _dataclass_to_dict(result)}

            elif cmd == "complete":
                result = worker.complete(
                    code=request.get("code", ""),
                    cursor_pos=request.get("cursor", 0),
                )
                response = {"type": "complete", "result": result}

            elif cmd == "inspect":
                result = worker.inspect(
                    code=request.get("code", ""),
                    cursor_pos=request.get("cursor", 0),
                    detail=request.get("detail", 1),
                )
                response = {"type": "inspect", "result": result}

            elif cmd == "hover":
                result = worker.hover(
                    code=request.get("code", ""),
                    cursor_pos=request.get("cursor", 0),
                )
                response = {"type": "hover", "result": result}

            elif cmd == "variables":
                result = worker.get_variables()
                response = {"type": "variables", "result": result}

            elif cmd == "reset":
                worker.reset()
                response = {"type": "reset", "success": True}

            elif cmd == "is_complete":
                result = worker.is_complete(code=request.get("code", ""))
                response = {"type": "is_complete", "result": result}

            elif cmd == "shutdown":
                response = {"type": "shutdown", "success": True}
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                break

            elif cmd == "ping":
                response = {"type": "pong"}

            else:
                response = {"type": "error", "error": f"Unknown command: {cmd}"}

        except Exception as e:
            tb = traceback.format_exc()
            response = {"type": "error", "error": str(e), "traceback": tb}

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
