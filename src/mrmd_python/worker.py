"""
IPython Worker

Core execution engine for Python code. Ported from mrmd prototype.

Features:
- PTY-based streaming for ANSI color support
- Matplotlib hook for saving figures as assets
- Rich display capture (PNG, HTML, SVG)
- Plotly renderer configuration
- Auto-reload for module development
- Variable inspection with drill-down
"""

import sys
import os
import io
import traceback
import signal
import threading
import select
import time
import base64
import re
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, field, asdict

from .types import (
    ExecuteResult,
    ExecuteError,
    Asset,
    DisplayData,
    CompleteResult,
    CompletionItem,
    InspectResult,
    HoverResult,
    Variable,
    VariablesResult,
    VariableDetail,
    IsCompleteResult,
    StdinRequest,
    InputCancelledError,
)


@dataclass
class StreamingExecuteResult:
    """Result from streaming execution"""

    success: bool = True
    stdout: str = ""
    stderr: str = ""
    result: str | None = None
    error: ExecuteError | None = None
    display_data: list[dict] = field(default_factory=list)
    assets: list[Asset] = field(default_factory=list)
    execution_count: int = 0
    duration: int | None = None


class IPythonWorker:
    """
    IPython worker that executes code and provides IDE features.

    Designed to be used by an HTTP server, not as a subprocess.
    When a custom venv is specified, execution happens via subprocess
    using the venv's Python interpreter.
    """

    def __init__(self, cwd: str | None = None, assets_dir: str | None = None, venv: str | None = None):
        self.cwd = cwd
        self.assets_dir = assets_dir
        self.venv = venv
        self.shell = None
        self._initialized = False
        self._captured_displays: list[dict] = []
        self._asset_counter = 0
        self._interrupt_requested = False
        self._current_exec_id: str | None = None
        self._pending_input: dict | None = None  # For stdin_request handling
        self._input_event: threading.Event | None = None
        self._input_response: str | None = None
        self._execution_count = 0
        self._subprocess_globals: dict = {}  # Simulated globals for subprocess mode

    def _get_venv_python(self) -> str | None:
        """Get the Python executable path for the configured venv."""
        if not self.venv:
            return None
        venv_path = Path(self.venv)
        if sys.platform == 'win32':
            python_exe = venv_path / 'Scripts' / 'python.exe'
        else:
            python_exe = venv_path / 'bin' / 'python'
        return str(python_exe) if python_exe.exists() else None

    def _ensure_initialized(self):
        """Lazy initialization of IPython shell."""
        if self._initialized:
            return

        from IPython.core.interactiveshell import InteractiveShell

        # Create shell instance
        self.shell = InteractiveShell.instance()

        # Enable rich display for return values
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

        # Set up Plotly for HTML output
        self._setup_plotly_renderer()

        # Register %pip magic with uv
        self._register_uv_pip_magic()

        # Change to working directory if specified
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

    def _save_asset(
        self, content: bytes, mime_type: str, extension: str
    ) -> Asset | None:
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

        # Determine asset type
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

    def _save_text_asset(
        self, content: str, mime_type: str, extension: str
    ) -> Asset | None:
        """Save text content as an asset file."""
        return self._save_asset(content.encode("utf-8"), mime_type, extension)

    def _wrap_html_standalone(self, html_content: str, title: str = "Output") -> str:
        """Wrap HTML content in a standalone document."""
        is_plotly = "plotly" in html_content.lower() or "Plotly.newPlot" in html_content

        if is_plotly:
            # Strip require.js wrapper if present
            if "require(" in html_content or "requirejs(" in html_content:
                html_content = re.sub(
                    r"require\s*\(\s*\[[^\]]*\]\s*,\s*function\s*\([^)]*\)\s*\{",
                    "(function() {",
                    html_content,
                )
                html_content = re.sub(
                    r"requirejs\s*\(\s*\[[^\]]*\]\s*,\s*function\s*\([^)]*\)\s*\{",
                    "(function() {",
                    html_content,
                )
                html_content = re.sub(
                    r"\}\s*\)\s*;?\s*</script>", "})();</script>", html_content
                )

            return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
    <style>
        body {{ margin: 0; padding: 8px; background: white; }}
        .plotly-graph-div {{ width: 100% !important; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
        else:
            return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 16px;
            background: white;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background: #f5f5f5;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background: #fafafa;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

    def _setup_display_capture(self):
        """Set up capture of rich display outputs with asset-based storage."""
        worker = self

        class AssetCapturingDisplayPublisher:
            def __init__(self, shell):
                self.shell = shell
                self.is_publishing = False

            def publish(self, data, metadata=None, source=None, **kwargs):
                self.is_publishing = True
                try:
                    self._process_display(data, metadata or {})
                finally:
                    self.is_publishing = False

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
                        worker._captured_displays.append(
                            {"data": data, "metadata": metadata}
                        )
                    return

                # Handle image/jpeg
                if "image/jpeg" in data:
                    jpeg_data = data["image/jpeg"]
                    if isinstance(jpeg_data, str):
                        jpeg_bytes = base64.b64decode(jpeg_data)
                    else:
                        jpeg_bytes = jpeg_data

                    asset = worker._save_asset(jpeg_bytes, "image/jpeg", "jpg")
                    if asset:
                        worker._captured_displays.append({"asset": asdict(asset)})
                    else:
                        worker._captured_displays.append(
                            {"data": data, "metadata": metadata}
                        )
                    return

                # Handle image/svg+xml
                if "image/svg+xml" in data:
                    svg_content = data["image/svg+xml"]
                    if isinstance(svg_content, bytes):
                        svg_content = svg_content.decode("utf-8")

                    asset = worker._save_text_asset(svg_content, "image/svg+xml", "svg")
                    if asset:
                        worker._captured_displays.append({"asset": asdict(asset)})
                    else:
                        worker._captured_displays.append(
                            {"data": data, "metadata": metadata}
                        )
                    return

                # Handle text/html
                if "text/html" in data:
                    html_content = data["text/html"]
                    if isinstance(html_content, bytes):
                        html_content = html_content.decode("utf-8")

                    if not html_content or not html_content.strip():
                        return

                    if len(html_content.strip()) < 50 and "<" not in html_content[10:]:
                        if "text/plain" in data:
                            worker._captured_displays.append(
                                {
                                    "data": {"text/plain": data["text/plain"]},
                                    "metadata": metadata,
                                }
                            )
                        return

                    # Skip Plotly config HTML
                    is_plotly_related = "plotly" in html_content.lower()
                    has_plot_content = (
                        "Plotly.newPlot" in html_content
                        or "plotly-graph-div" in html_content
                    )
                    if is_plotly_related and not has_plot_content:
                        return

                    standalone_html = worker._wrap_html_standalone(html_content)
                    asset = worker._save_text_asset(
                        standalone_html, "text/html", "html"
                    )

                    if asset:
                        worker._captured_displays.append({"asset": asdict(asset)})
                    else:
                        worker._captured_displays.append(
                            {"data": data, "metadata": metadata}
                        )
                    return

                # Handle text/plain and other types
                if "text/plain" in data:
                    worker._captured_displays.append(
                        {"data": {"text/plain": data["text/plain"]}, "metadata": metadata}
                    )
                    return

                if data:
                    worker._captured_displays.append(
                        {"data": data, "metadata": metadata}
                    )

            def clear_output(self, wait=False):
                pass

        self.shell.display_pub = AssetCapturingDisplayPublisher(self.shell)

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

    def _register_uv_pip_magic(self):
        """Register %pip magic that uses uv."""
        import subprocess
        import shutil

        python_executable = sys.executable
        cwd = self.cwd

        def pip_magic(line):
            uv_path = shutil.which("uv")
            if not uv_path:
                print("Error: uv not found. Install from https://docs.astral.sh/uv/")
                return

            args = line.split()
            if args:
                subcommand = args[0]
                rest = args[1:]
                cmd = [uv_path, "pip", subcommand, "-p", python_executable] + rest
            else:
                cmd = [uv_path, "pip"]

            print(f"Running: uv pip {' '.join(args)}")
            print("-" * 50)

            try:
                subprocess.run(cmd, capture_output=False, text=True, cwd=cwd)
            except Exception as e:
                print(f"Error: {e}")

            print("-" * 50)
            print("Note: Restart kernel to use newly installed packages.")

        self.shell.register_magic_function(pip_magic, "line", "pip")

        def add_magic(line):
            """Add packages using uv add."""
            uv_path = shutil.which("uv")
            if not uv_path:
                print("Error: uv not found.")
                return

            if not line.strip():
                print("Usage: %add package1 package2 ...")
                return

            project_dir = Path(cwd) if cwd else Path.cwd()
            if not (project_dir / "pyproject.toml").exists():
                print("No pyproject.toml found. Use %pip install instead.")
                return

            packages = line.split()
            cmd = [uv_path, "add"] + packages

            print(f"Adding: {', '.join(packages)}")
            print("-" * 50)

            try:
                result = subprocess.run(
                    cmd, capture_output=False, text=True, cwd=str(project_dir)
                )
                if result.returncode == 0:
                    print("-" * 50)
                    print("Packages added. Restart kernel to use.")
            except Exception as e:
                print(f"Error: {e}")

        self.shell.register_magic_function(add_magic, "line", "add")

    def _setup_autoreload(self):
        """Enable autoreload extension."""
        try:
            self.shell.run_line_magic("load_ext", "autoreload")
            self.shell.run_line_magic("autoreload", "2")
        except Exception:
            pass

    # =========================================================================
    # Execution
    # =========================================================================

    def _execute_subprocess(
        self, code: str, exec_id: str | None = None
    ) -> ExecuteResult:
        """Execute code in a subprocess using the configured venv's Python."""
        import subprocess
        import json

        python_exe = self._get_venv_python()
        if not python_exe:
            return ExecuteResult(
                success=False,
                error=ExecuteError(
                    ename="VenvError",
                    evalue="Could not find Python executable in venv",
                    traceback=[]
                )
            )

        self._execution_count += 1
        start_time = time.time()

        # Create a wrapper script that executes the code and captures output
        wrapper_code = '''
import sys
import json

# Execute the user's code
_mrmd_result = None
_mrmd_error = None
_mrmd_stdout = ""

try:
    exec(compile("""''' + code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"') + '''""", "<cell>", "exec"))
except Exception as e:
    import traceback
    _mrmd_error = {
        "ename": type(e).__name__,
        "evalue": str(e),
        "traceback": traceback.format_exception(type(e), e, e.__traceback__)
    }
'''

        try:
            result = subprocess.run(
                [python_exe, "-c", wrapper_code],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=self.cwd,
            )

            duration = int((time.time() - start_time) * 1000)

            if result.returncode == 0:
                return ExecuteResult(
                    success=True,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    executionCount=self._execution_count,
                    duration=duration,
                )
            else:
                # Parse error from stderr if possible
                error_lines = result.stderr.strip().split('\n') if result.stderr else []
                return ExecuteResult(
                    success=False,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    error=ExecuteError(
                        ename="ExecutionError",
                        evalue=error_lines[-1] if error_lines else "Unknown error",
                        traceback=error_lines
                    ),
                    executionCount=self._execution_count,
                    duration=duration,
                )

        except subprocess.TimeoutExpired:
            return ExecuteResult(
                success=False,
                error=ExecuteError(
                    ename="TimeoutError",
                    evalue="Execution timed out after 5 minutes",
                    traceback=[]
                ),
                executionCount=self._execution_count,
            )
        except Exception as e:
            return ExecuteResult(
                success=False,
                error=ExecuteError(
                    ename=type(e).__name__,
                    evalue=str(e),
                    traceback=[]
                ),
                executionCount=self._execution_count,
            )

    def _execute_subprocess_streaming(
        self,
        code: str,
        on_output: Callable[[str, str, str], None],
        exec_id: str | None = None,
    ) -> ExecuteResult:
        """Execute code in subprocess with streaming output."""
        import subprocess

        python_exe = self._get_venv_python()
        if not python_exe:
            return ExecuteResult(
                success=False,
                error=ExecuteError(
                    ename="VenvError",
                    evalue="Could not find Python executable in venv",
                    traceback=[]
                )
            )

        self._execution_count += 1
        start_time = time.time()
        accumulated_stdout = ""
        accumulated_stderr = ""

        try:
            # Start subprocess
            process = subprocess.Popen(
                [python_exe, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.cwd,
            )

            # Read output in real-time
            import selectors
            sel = selectors.DefaultSelector()
            sel.register(process.stdout, selectors.EVENT_READ)
            sel.register(process.stderr, selectors.EVENT_READ)

            while process.poll() is None or sel.get_map():
                for key, _ in sel.select(timeout=0.1):
                    data = key.fileobj.read(1)
                    if not data:
                        sel.unregister(key.fileobj)
                        continue
                    if key.fileobj == process.stdout:
                        accumulated_stdout += data
                        on_output("stdout", data, accumulated_stdout)
                    else:
                        accumulated_stderr += data
                        on_output("stderr", data, accumulated_stderr)

            # Read any remaining output
            remaining_stdout = process.stdout.read()
            remaining_stderr = process.stderr.read()
            if remaining_stdout:
                accumulated_stdout += remaining_stdout
                on_output("stdout", remaining_stdout, accumulated_stdout)
            if remaining_stderr:
                accumulated_stderr += remaining_stderr
                on_output("stderr", remaining_stderr, accumulated_stderr)

            duration = int((time.time() - start_time) * 1000)
            success = process.returncode == 0

            if not success:
                error_lines = accumulated_stderr.strip().split('\n') if accumulated_stderr else []
                return ExecuteResult(
                    success=False,
                    stdout=accumulated_stdout,
                    stderr=accumulated_stderr,
                    error=ExecuteError(
                        ename="ExecutionError",
                        evalue=error_lines[-1] if error_lines else "Unknown error",
                        traceback=error_lines
                    ),
                    executionCount=self._execution_count,
                    duration=duration,
                )

            return ExecuteResult(
                success=True,
                stdout=accumulated_stdout,
                stderr=accumulated_stderr,
                executionCount=self._execution_count,
                duration=duration,
            )

        except Exception as e:
            return ExecuteResult(
                success=False,
                error=ExecuteError(
                    ename=type(e).__name__,
                    evalue=str(e),
                    traceback=[]
                ),
                executionCount=self._execution_count,
            )

    def execute(
        self, code: str, store_history: bool = True, exec_id: str | None = None
    ) -> ExecuteResult:
        """Execute code and return result (non-streaming)."""
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
            exec_result = self.shell.run_cell(code, store_history=store_history, silent=False)
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
                    result.displayData.append(
                        DisplayData(data=disp["data"], metadata=disp.get("metadata", {}))
                    )

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
        on_output: Callable[[str, str, str], None],
        store_history: bool = True,
        exec_id: str | None = None,
        on_stdin_request: Callable[[StdinRequest], str] | None = None,
    ) -> ExecuteResult:
        """
        Execute code with streaming output and optional stdin support.

        Args:
            code: Code to execute
            on_output: Callback(stream, content, accumulated) for each output chunk
            store_history: Whether to store in IPython history
            exec_id: Execution ID for asset naming
            on_stdin_request: Callback(StdinRequest) -> str for handling input() calls.
                              If None, input() will raise an error.

        Returns:
            ExecuteResult with final result
        """
        self._ensure_initialized()
        self._captured_displays = []
        self._current_exec_id = exec_id
        self._asset_counter = 0

        # Try to use PTY for TTY emulation (enables ANSI colors)
        use_pty = False
        try:
            import pty

            stdout_master_fd, stdout_slave_fd = pty.openpty()
            stderr_master_fd, stderr_slave_fd = pty.openpty()
            use_pty = True
        except (ImportError, OSError):
            stdout_master_fd, stdout_slave_fd = os.pipe()
            stderr_master_fd, stderr_slave_fd = os.pipe()

        # Save original FDs
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)

        # Redirect stdout/stderr
        os.dup2(stdout_slave_fd, 1)
        os.dup2(stderr_slave_fd, 2)
        os.close(stdout_slave_fd)
        os.close(stderr_slave_fd)

        # Make master FDs non-blocking
        try:
            import fcntl

            for fd in [stdout_master_fd, stderr_master_fd]:
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except ImportError:
            pass

        accumulated_stdout = []
        accumulated_stderr = []
        stop_reader = threading.Event()
        start_time = time.time()

        def reader_thread():
            while not stop_reader.is_set():
                try:
                    readable, _, _ = select.select(
                        [stdout_master_fd, stderr_master_fd], [], [], 0.05
                    )
                except (ValueError, OSError):
                    break

                for fd in readable:
                    try:
                        data = os.read(fd, 4096)
                        if not data:
                            continue

                        if fd == stdout_master_fd:
                            stream_name = "stdout"
                            buffer_list = accumulated_stdout
                        else:
                            stream_name = "stderr"
                            buffer_list = accumulated_stderr

                        text = data.decode("utf-8", errors="replace")
                        buffer_list.append(text)
                        on_output(stream_name, text, "".join(buffer_list))

                    except (BlockingIOError, OSError):
                        pass

            # Final read
            for fd, stream_name, buffer_list in [
                (stdout_master_fd, "stdout", accumulated_stdout),
                (stderr_master_fd, "stderr", accumulated_stderr),
            ]:
                try:
                    while True:
                        data = os.read(fd, 4096)
                        if not data:
                            break
                        text = data.decode("utf-8", errors="replace")
                        buffer_list.append(text)
                        on_output(stream_name, text, "".join(buffer_list))
                except (BlockingIOError, OSError):
                    pass

        reader = threading.Thread(target=reader_thread, daemon=True)
        reader.start()

        # Update sys.stdout/stderr
        sys.stdout = io.TextIOWrapper(
            io.FileIO(1, mode="w", closefd=False), line_buffering=True
        )
        sys.stderr = io.TextIOWrapper(
            io.FileIO(2, mode="w", closefd=False), line_buffering=True
        )

        result = ExecuteResult()

        # Hook builtins.input for stdin support
        import builtins
        import getpass as getpass_module
        original_input = builtins.input
        original_getpass = getpass_module.getpass

        def hooked_input(prompt=""):
            """Custom input() that uses the stdin callback."""
            if on_stdin_request is None:
                raise RuntimeError(
                    "input() is not supported in this execution context. "
                    "The client must provide stdin support."
                )

            # Flush stdout so prompt appears before we request input
            sys.stdout.write(prompt)
            sys.stdout.flush()

            # Create stdin request
            request = StdinRequest(
                prompt=prompt,
                password=False,
                execId=exec_id or ""
            )

            # Call the callback and wait for response
            # The callback is expected to block until input is provided
            response = on_stdin_request(request)

            # Echo the input (like a terminal would)
            sys.stdout.write(response)
            if not response.endswith('\n'):
                sys.stdout.write('\n')
            sys.stdout.flush()

            # Return without trailing newline (like real input())
            return response.rstrip('\n')

        def hooked_getpass(prompt="Password: ", stream=None):
            """Custom getpass() that uses the stdin callback with password=True."""
            if on_stdin_request is None:
                raise RuntimeError(
                    "getpass() is not supported in this execution context. "
                    "The client must provide stdin support."
                )

            # Flush stdout so prompt appears before we request input
            sys.stdout.write(prompt)
            sys.stdout.flush()

            # Create stdin request with password=True
            request = StdinRequest(
                prompt=prompt,
                password=True,
                execId=exec_id or ""
            )

            # Call the callback and wait for response
            response = on_stdin_request(request)

            # DON'T echo password input (unlike regular input)
            # Just write a newline to move to next line
            sys.stdout.write('\n')
            sys.stdout.flush()

            # Return without trailing newline
            return response.rstrip('\n')

        builtins.input = hooked_input
        getpass_module.getpass = hooked_getpass

        try:
            exec_result = self.shell.run_cell(code, store_history=store_history, silent=False)
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
                    result.displayData.append(
                        DisplayData(data=disp["data"], metadata=disp.get("metadata", {}))
                    )

        except KeyboardInterrupt:
            result.error = ExecuteError(
                type="KeyboardInterrupt", message="Interrupted"
            )
            result.success = False
        except InputCancelledError:
            # User cancelled input - not an error, just stop execution cleanly
            result.error = ExecuteError(
                type="InputCancelled", message="Input cancelled by user"
            )
            result.success = False
        except Exception as e:
            # Check if this is an InputCancelledError wrapped in another exception
            if "Input cancelled" in str(e) or isinstance(e.__cause__, InputCancelledError):
                result.error = ExecuteError(
                    type="InputCancelled", message="Input cancelled by user"
                )
            else:
                result.error = self._format_exception(e)
            result.success = False

        finally:
            # Restore original input and getpass functions
            builtins.input = original_input
            getpass_module.getpass = original_getpass

            sys.stdout.flush()
            sys.stderr.flush()

            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

            stop_reader.set()
            reader.join(timeout=1.0)

            os.close(stdout_master_fd)
            os.close(stderr_master_fd)

            sys.stdout = io.TextIOWrapper(
                io.FileIO(1, mode="w", closefd=False), line_buffering=True
            )
            sys.stderr = io.TextIOWrapper(
                io.FileIO(2, mode="w", closefd=False), line_buffering=True
            )

            result.stdout = "".join(accumulated_stdout)
            result.stderr = "".join(accumulated_stderr)
            result.duration = int((time.time() - start_time) * 1000)
            self._current_exec_id = None

        return result

    # =========================================================================
    # Completion
    # =========================================================================

    def complete(self, code: str, cursor_pos: int) -> CompleteResult:
        """Get completions at cursor position."""
        self._ensure_initialized()

        try:
            from IPython.core.completer import provisionalcompleter

            with provisionalcompleter():
                completions = list(self.shell.Completer.completions(code, cursor_pos))

            if completions:
                items = []
                for c in completions:
                    kind = self._map_completion_type(c.type) if c.type else "variable"
                    items.append(
                        CompletionItem(
                            label=c.text,
                            kind=kind,
                            type=c.type,
                        )
                    )
                return CompleteResult(
                    matches=items,
                    cursorStart=completions[0].start,
                    cursorEnd=completions[0].end,
                    source="runtime",
                )
        except Exception:
            pass

        return CompleteResult(cursorStart=cursor_pos, cursorEnd=cursor_pos)

    def _map_completion_type(self, type_str: str | None) -> str:
        """Map IPython completion type to MRP kind."""
        if not type_str:
            return "variable"
        mapping = {
            "function": "function",
            "builtin_function_or_method": "function",
            "method": "method",
            "class": "class",
            "module": "module",
            "keyword": "keyword",
            "instance": "variable",
            "statement": "keyword",
        }
        return mapping.get(type_str, "variable")

    # =========================================================================
    # Inspection
    # =========================================================================

    def inspect(self, code: str, cursor_pos: int, detail: int = 1) -> InspectResult:
        """Get detailed info about symbol at cursor."""
        self._ensure_initialized()

        try:
            name = self._extract_name_at_cursor(code, cursor_pos)
            if not name:
                return InspectResult(found=False)

            info = self.shell.object_inspect(name)
            if not info.get("found"):
                return InspectResult(found=False, name=name)

            return InspectResult(
                found=True,
                source="runtime",
                name=name,
                kind=self._classify_object_kind(info),
                type=info.get("type_name"),
                signature=info.get("call_signature") or info.get("init_signature"),
                docstring=info.get("docstring") if detail >= 1 else None,
                sourceCode=info.get("source") if detail >= 2 else None,
                file=info.get("file"),
                line=info.get("line"),
            )
        except Exception:
            return InspectResult(found=False)

    def hover(self, code: str, cursor_pos: int) -> HoverResult:
        """Get hover tooltip for symbol."""
        self._ensure_initialized()

        try:
            name = self._extract_name_at_cursor(code, cursor_pos)
            if not name:
                return HoverResult(found=False)

            # Try to get value from namespace
            try:
                value = eval(name, {"__builtins__": {}}, self.shell.user_ns)
                type_name = type(value).__name__

                # Get value preview
                value_str = self._get_value_preview(value)

                # Get signature if callable
                signature = None
                if callable(value):
                    try:
                        import inspect
                        sig = inspect.signature(value)
                        signature = f"{name}{sig}"
                    except Exception:
                        pass

                return HoverResult(
                    found=True,
                    name=name,
                    type=type_name,
                    value=value_str,
                    signature=signature,
                )
            except Exception:
                pass

            # Fall back to object_inspect
            info = self.shell.object_inspect(name)
            if info.get("found"):
                return HoverResult(
                    found=True,
                    name=name,
                    type=info.get("type_name"),
                    signature=info.get("call_signature"),
                )

            return HoverResult(found=False)
        except Exception:
            return HoverResult(found=False)

    def _get_value_preview(self, value: Any, max_len: int = 200) -> str:
        """Get a preview string of a value."""
        try:
            if hasattr(value, "shape") and hasattr(value, "head"):
                # DataFrame
                if hasattr(value, "columns"):
                    return f"DataFrame {value.shape}\n{value.head(3).to_string()}"
                else:
                    return f"Series {value.shape}\n{value.head(5).to_string()}"
            elif hasattr(value, "shape") and hasattr(value, "dtype"):
                # numpy array
                flat = value.flatten()[:10]
                preview = ", ".join(str(x) for x in flat)
                if len(value.flatten()) > 10:
                    preview += ", ..."
                return f"array{value.shape} dtype={value.dtype}\n[{preview}]"
            elif isinstance(value, (list, tuple)):
                preview = repr(value)
                if len(preview) > max_len:
                    preview = preview[:max_len] + "..."
                return f"{type(value).__name__}({len(value)} items)\n{preview}"
            elif isinstance(value, dict):
                preview = repr(value)
                if len(preview) > max_len:
                    preview = preview[:max_len] + "..."
                return f"dict({len(value)} keys)\n{preview}"
            else:
                preview = repr(value)
                if len(preview) > max_len:
                    preview = preview[:max_len] + "..."
                return preview
        except Exception:
            return f"<{type(value).__name__}>"

    def _classify_object_kind(self, info: dict) -> str:
        """Classify object kind from inspect info."""
        type_name = info.get("type_name", "")
        if type_name in ("function", "builtin_function_or_method"):
            return "function"
        elif type_name == "type":
            return "class"
        elif type_name == "module":
            return "module"
        elif type_name == "method":
            return "method"
        elif info.get("call_signature"):
            return "function"
        return "variable"

    # =========================================================================
    # Variables
    # =========================================================================

    def get_variables(self) -> VariablesResult:
        """Get user variables with type info."""
        self._ensure_initialized()

        variables = []
        user_ns = self.shell.user_ns

        skip_prefixes = ("_", "In", "Out", "get_ipython", "exit", "quit")
        skip_types = (type(sys),)

        import builtins

        builtin_names = set(dir(builtins))

        for name, value in user_ns.items():
            if name.startswith(skip_prefixes):
                continue
            if isinstance(value, skip_types):
                continue
            if name in builtin_names:
                continue
            if (
                callable(value)
                and hasattr(value, "__module__")
                and value.__module__ == "builtins"
            ):
                continue

            try:
                var = self._make_variable(name, value)
                variables.append(var)
            except Exception:
                pass

        variables.sort(key=lambda v: v.name)
        return VariablesResult(
            variables=variables, count=len(variables), truncated=False
        )

    def get_variable_detail(
        self, name: str, path: list[str] | None = None
    ) -> VariableDetail:
        """Get detailed info about a variable."""
        self._ensure_initialized()

        try:
            # Navigate path
            full_path = name
            if path:
                full_path = f"{name}.{'.'.join(path)}"

            value = eval(full_path, {"__builtins__": {}}, self.shell.user_ns)
            var = self._make_variable(name, value)

            # Get children
            children = []
            if isinstance(value, dict):
                for k, v in list(value.items())[:100]:
                    children.append(self._make_variable(repr(k), v))
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(list(value)[:100]):
                    children.append(self._make_variable(f"[{i}]", v))
            elif hasattr(value, "__dict__"):
                for k, v in list(vars(value).items())[:100]:
                    if not k.startswith("_"):
                        children.append(self._make_variable(k, v))

            return VariableDetail(
                name=var.name,
                type=var.type,
                value=var.value,
                size=var.size,
                expandable=var.expandable,
                shape=var.shape,
                dtype=var.dtype,
                length=var.length,
                children=children if children else None,
            )
        except Exception as e:
            return VariableDetail(name=name, type="error", value=str(e))

    def _make_variable(self, name: str, value: Any) -> Variable:
        """Create a Variable from a value."""
        type_name = type(value).__name__
        expandable = False
        shape = None
        dtype = None
        length = None
        size = None

        # Get shape/size
        if hasattr(value, "shape"):
            shape = list(value.shape)
            expandable = True
            if hasattr(value, "dtype"):
                dtype = str(value.dtype)
            if hasattr(value, "columns"):
                size = f"{value.shape[0]} rows × {value.shape[1]} cols"
        elif isinstance(value, dict):
            length = len(value)
            size = f"{length} items"
            expandable = length > 0
        elif isinstance(value, (list, tuple, set, frozenset)):
            length = len(value)
            size = f"{length} items"
            expandable = length > 0
        elif isinstance(value, str):
            length = len(value)
            expandable = length > 50
        elif hasattr(value, "__dict__"):
            expandable = True

        # Get value preview
        try:
            preview = repr(value)
            if len(preview) > 80:
                preview = preview[:77] + "..."
        except Exception:
            preview = f"<{type_name}>"

        return Variable(
            name=name,
            type=type_name,
            value=preview,
            size=size,
            expandable=expandable,
            shape=shape,
            dtype=dtype,
            length=length,
        )

    # =========================================================================
    # Code Analysis
    # =========================================================================

    def is_complete(self, code: str) -> IsCompleteResult:
        """Check if code is a complete statement."""
        self._ensure_initialized()

        try:
            status, indent = self.shell.input_transformer_manager.check_complete(code)
            return IsCompleteResult(status=status, indent=indent or "")
        except Exception:
            return IsCompleteResult(status="unknown")

    def format_code(self, code: str) -> tuple[str, bool]:
        """Format code using black."""
        try:
            import black

            formatted = black.format_str(code, mode=black.Mode())
            return formatted, formatted != code
        except ImportError:
            return code, False
        except Exception:
            return code, False

    # =========================================================================
    # Session Management
    # =========================================================================

    def reset(self):
        """Reset the namespace."""
        self._ensure_initialized()
        self.shell.reset()

    def get_info(self) -> dict:
        """Get info about this worker."""
        return {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "cwd": os.getcwd(),
        }

    # =========================================================================
    # Helpers
    # =========================================================================

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
