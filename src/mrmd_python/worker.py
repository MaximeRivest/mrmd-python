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
import uuid
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
    HistoryEntry,
    HistoryResult,
    StdinRequest,
    InputCancelledError,
    StdinNotAllowedError,
)
from .subprocess_manager import SubprocessWorker


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
        self._executing_thread_id: int | None = None  # Thread ID for interrupt

        # Set matplotlib backend to Agg early (before any imports)
        # This ensures plots work even if matplotlib is installed later via %pip
        os.environ.setdefault("MPLBACKEND", "Agg")
        self._current_exec_id: str | None = None
        self._pending_input: dict | None = None  # For stdin_request handling
        self._input_event: threading.Event | None = None
        self._input_response: str | None = None
        self._execution_count = 0
        self._subprocess_worker: SubprocessWorker | None = None  # Persistent subprocess for different venv

    def _get_venv_python(self) -> str | None:
        """Get the Python executable path for the configured venv."""
        if not self.venv:
            return None
        venv_path = Path(self.venv)
        if sys.platform == 'win32':
            candidates = [venv_path / 'Scripts' / 'python.exe']
        else:
            candidates = [
                venv_path / 'bin' / 'python',
                venv_path / 'bin' / 'python3',
            ]

        for python_exe in candidates:
            if python_exe.exists():
                return str(python_exe)

        return None

    def _should_use_subprocess(self) -> bool:
        """Check if we should use subprocess for execution (different venv).

        Returns True if:
        - A venv is configured
        - The venv differs from the current Python's prefix
        - The venv's Python executable exists
        """
        if not self.venv:
            return False

        # Compare venv path to current Python's prefix (use realpath to handle symlinks)
        current_prefix = os.path.realpath(sys.prefix)
        target_venv = os.path.realpath(self.venv)

        if current_prefix == target_venv:
            return False

        # Verify the target Python exists
        return self._get_venv_python() is not None

    def _ensure_mrmd_python_in_venv(self) -> bool:
        """Ensure mrmd-python is installed in the target venv.

        Uses uv pip install to add mrmd-python to the venv so that
        subprocess_worker can be invoked with -m mrmd_python.subprocess_worker.

        Returns True if successful.
        """
        import subprocess as sp

        python_exe = self._get_venv_python()
        if not python_exe:
            return False

        # Check if mrmd_python is already importable
        check_result = sp.run(
            [python_exe, "-c", "import mrmd_python"],
            capture_output=True,
            text=True,
        )

        if check_result.returncode == 0:
            # Already installed
            return True

        # Install mrmd-python using uv pip
        try:
            install_result = sp.run(
                ["uv", "pip", "install", "--python", python_exe, "mrmd-python"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return install_result.returncode == 0
        except (FileNotFoundError, sp.TimeoutExpired):
            # uv not available or timeout - try pip directly
            try:
                install_result = sp.run(
                    [python_exe, "-m", "pip", "install", "mrmd-python"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                return install_result.returncode == 0
            except (FileNotFoundError, sp.TimeoutExpired):
                return False

    def _get_subprocess_worker(self) -> SubprocessWorker:
        """Get or create the subprocess worker for different-venv execution."""
        if self._subprocess_worker is None or not self._subprocess_worker.is_alive():
            # Ensure mrmd-python is installed in the target venv
            if not self._ensure_mrmd_python_in_venv():
                raise RuntimeError(
                    f"Could not install mrmd-python in venv {self.venv}. "
                    "Please install it manually: uv pip install mrmd-python"
                )

            self._subprocess_worker = SubprocessWorker(
                venv=self.venv,
                cwd=self.cwd,
                assets_dir=self.assets_dir,
            )
        return self._subprocess_worker

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

    def _new_asset_id(self, extension: str) -> str:
        """Create a server-generated opaque asset ID."""
        return f"asset-{uuid.uuid4().hex[:16]}.{extension}"

    def _save_asset(
        self, content: bytes, mime_type: str, extension: str
    ) -> Asset | None:
        """Save content as an asset file."""
        if not self.assets_dir:
            return None

        assets_path = Path(self.assets_dir)
        assets_path.mkdir(parents=True, exist_ok=True)

        asset_id = self._new_asset_id(extension)
        filepath = assets_path / asset_id
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
            id=asset_id,
            url=f"/mrp/v1/assets/{asset_id}",
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

                        asset_id = worker._new_asset_id("png")
                        filepath = assets_path / asset_id
                        filepath.write_bytes(png_bytes)

                        asset = Asset(
                            id=asset_id,
                            url=f"/mrp/v1/assets/{asset_id}",
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

                    asset_id = worker._new_asset_id("png")
                    filepath = assets_path / asset_id
                    filepath.write_bytes(png_bytes)

                    asset = Asset(
                        id=asset_id,
                        url=f"/mrp/v1/assets/{asset_id}",
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
        self,
        code: str,
        exec_id: str | None = None,
        store_history: bool = True,
        allow_stdin: bool = False,
    ) -> ExecuteResult:
        """Execute code using the persistent SubprocessWorker (IPython-based)."""
        try:
            worker = self._get_subprocess_worker()
            return worker.execute(
                code,
                store_history=store_history,
                exec_id=exec_id,
                allow_stdin=allow_stdin,
            )
        except Exception as e:
            return ExecuteResult(
                success=False,
                error=ExecuteError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=[]
                ),
            )

    def _execute_subprocess_streaming(
        self,
        code: str,
        on_output: Callable[[str, str, str], None],
        exec_id: str | None = None,
        store_history: bool = True,
        allow_stdin: bool = True,
        on_stdin_request: Callable[[StdinRequest], str] | None = None,
    ) -> ExecuteResult:
        """Execute code using the persistent SubprocessWorker with streaming."""
        try:
            worker = self._get_subprocess_worker()
            return worker.execute_streaming(
                code,
                on_output=on_output,
                store_history=store_history,
                exec_id=exec_id,
                allow_stdin=allow_stdin,
                on_stdin_request=on_stdin_request,
            )
        except Exception as e:
            return ExecuteResult(
                success=False,
                error=ExecuteError(
                    type=type(e).__name__,
                    message=str(e),
                    traceback=[]
                ),
            )

    def execute(
        self,
        code: str,
        store_history: bool = True,
        exec_id: str | None = None,
        allow_stdin: bool = False,
    ) -> ExecuteResult:
        """Execute code and return result (non-streaming).

        Synchronous execution does not support interactive stdin round-trips.
        If code calls input()/getpass(), execution fails with StdinNotAllowed
        instead of hanging the runtime.
        """
        # Use subprocess execution if venv differs from current Python
        if self._should_use_subprocess():
            return self._execute_subprocess(
                code,
                exec_id,
                store_history=store_history,
                allow_stdin=allow_stdin,
            )

        self._ensure_initialized()
        self._captured_displays = []
        self._current_exec_id = exec_id
        self._asset_counter = 0

        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = captured_stdout = io.StringIO()
        sys.stderr = captured_stderr = io.StringIO()

        import builtins
        import getpass as getpass_module
        original_input = builtins.input
        original_getpass = getpass_module.getpass

        def disallow_stdin(*args, **kwargs):
            raise StdinNotAllowedError(
                "stdin is only supported with /execute/stream when the client handles input"
            )

        builtins.input = disallow_stdin
        getpass_module.getpass = disallow_stdin

        start_time = time.time()
        result = ExecuteResult()

        self._executing_thread_id = threading.get_ident()
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
                    result.warnings.append({
                        "code": "repr_failed",
                        "message": "Result repr failed; returned fallback preview",
                    })

            if exec_result.error_in_exec:
                if isinstance(exec_result.error_in_exec, StdinNotAllowedError):
                    result.error = ExecuteError(type="StdinNotAllowed", message=str(exec_result.error_in_exec))
                else:
                    result.error = self._format_exception(exec_result.error_in_exec)
                result.success = False
            elif exec_result.error_before_exec:
                if isinstance(exec_result.error_before_exec, StdinNotAllowedError):
                    result.error = ExecuteError(type="StdinNotAllowed", message=str(exec_result.error_before_exec))
                else:
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

        except StdinNotAllowedError as e:
            result.error = ExecuteError(type="StdinNotAllowed", message=str(e))
            result.success = False
        except Exception as e:
            result.error = self._format_exception(e)
            result.success = False

        finally:
            builtins.input = original_input
            getpass_module.getpass = original_getpass
            sys.stdout, sys.stderr = old_stdout, old_stderr
            result.stdout = captured_stdout.getvalue()
            result.stderr = captured_stderr.getvalue()
            result.duration = int((time.time() - start_time) * 1000)
            self._current_exec_id = None
            self._executing_thread_id = None

        return result

    def execute_streaming(
        self,
        code: str,
        on_output: Callable[[str, str, str], None],
        store_history: bool = True,
        exec_id: str | None = None,
        on_stdin_request: Callable[[StdinRequest], str] | None = None,
        allow_stdin: bool = True,
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
        # Use subprocess execution if venv differs from current Python
        if self._should_use_subprocess():
            return self._execute_subprocess_streaming(
                code,
                on_output,
                exec_id,
                store_history=store_history,
                allow_stdin=allow_stdin,
                on_stdin_request=on_stdin_request,
            )

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

        self._executing_thread_id = threading.get_ident()

        # Hook builtins.input for stdin support
        import builtins
        import getpass as getpass_module
        original_input = builtins.input
        original_getpass = getpass_module.getpass

        def hooked_input(prompt=""):
            """Custom input() that uses the stdin callback."""
            if on_stdin_request is None or not allow_stdin:
                raise StdinNotAllowedError(
                    "stdin is only supported when the client handles input"
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
            if on_stdin_request is None or not allow_stdin:
                raise StdinNotAllowedError(
                    "stdin is only supported when the client handles input"
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
                if isinstance(exec_result.error_in_exec, StdinNotAllowedError):
                    result.error = ExecuteError(type="StdinNotAllowed", message=str(exec_result.error_in_exec))
                else:
                    result.error = self._format_exception(exec_result.error_in_exec)
                result.success = False
            elif exec_result.error_before_exec:
                if isinstance(exec_result.error_before_exec, StdinNotAllowedError):
                    result.error = ExecuteError(type="StdinNotAllowed", message=str(exec_result.error_before_exec))
                else:
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
        except StdinNotAllowedError as e:
            result.error = ExecuteError(type="StdinNotAllowed", message=str(e))
            result.success = False
        except Exception as e:
            # Check if this is an InputCancelledError wrapped in another exception
            if "Input cancelled" in str(e) or isinstance(e.__cause__, InputCancelledError):
                result.error = ExecuteError(
                    type="InputCancelled", message="Input cancelled by user"
                )
            elif isinstance(e.__cause__, StdinNotAllowedError) or "stdin is only supported" in str(e):
                result.error = ExecuteError(type="StdinNotAllowed", message="stdin is only supported when the client handles input")
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
            self._executing_thread_id = None

        return result

    # =========================================================================
    # Completion
    # =========================================================================

    def complete(self, code: str, cursor_pos: int) -> CompleteResult:
        """Get completions at cursor position."""
        self._ensure_initialized()

        # Check if we're inside a function call — if so, inject
        # parameter name completions and filter out noise.
        # Done outside the try/except so an IPython crash doesn't lose params.
        param_items, call_info = self._param_completions(code, cursor_pos)

        try:
            from IPython.core.completer import provisionalcompleter

            # Clamp cursor to valid range for IPython
            safe_cursor = min(cursor_pos, len(code))

            with provisionalcompleter():
                completions = list(self.shell.Completer.completions(code, safe_cursor))

            if completions:
                items = []

                # When inside a function call, filter out filesystem paths
                # and irrelevant globals. Keep only: param names from
                # signature, plus completions that look like valid Python
                # identifiers (variables, functions, etc.)
                in_call = call_info is not None
                seen_labels = set()

                # Parameter completions go first
                for p in param_items:
                    seen_labels.add(p.label)
                    items.append(p)

                for c in completions:
                    if c.text in seen_labels:
                        continue

                    # Inside function calls, skip filesystem completions
                    # (contain /, ., or look like paths/filenames)
                    if in_call and self._looks_like_path(c.text):
                        continue

                    kind = self._map_completion_type(c.type) if c.type else "variable"
                    detail, documentation, sort_prefix = self._completion_detail(
                        c.text, c.type, code, cursor_pos
                    )
                    # When inside a call, push non-param completions below params
                    if in_call and param_items:
                        sort_prefix = "c" + sort_prefix

                    items.append(
                        CompletionItem(
                            label=c.text,
                            kind=kind,
                            type=c.type,
                            detail=detail,
                            documentation=documentation,
                            sortText=sort_prefix + c.text,
                        )
                    )

                cursor_start = completions[0].start if completions else cursor_pos
                cursor_end = completions[0].end if completions else cursor_pos

                # If we only have param completions (no IPython matches),
                # set the range to cover any partial text before cursor
                if not completions and param_items:
                    # Find where the current token starts
                    token_start = cursor_pos
                    while token_start > 0 and code[token_start - 1:token_start].isalnum() or (
                        token_start > 0 and code[token_start - 1:token_start] == '_'
                    ):
                        token_start -= 1
                    cursor_start = token_start
                    cursor_end = cursor_pos

                return CompleteResult(
                    matches=items,
                    cursorStart=cursor_start,
                    cursorEnd=cursor_end,
                    source="runtime",
                )
            elif param_items:
                # No IPython completions but we have param completions
                token_start = cursor_pos
                while token_start > 0 and (code[token_start - 1:token_start].isalnum()
                                           or code[token_start - 1:token_start] == '_'):
                    token_start -= 1
                return CompleteResult(
                    matches=param_items,
                    cursorStart=token_start,
                    cursorEnd=cursor_pos,
                    source="runtime",
                )
        except Exception:
            pass

        return CompleteResult(cursorStart=cursor_pos, cursorEnd=cursor_pos)

    def _param_completions(
        self, code: str, cursor_pos: int
    ) -> tuple[list[CompletionItem], dict | None]:
        """Detect if cursor is inside a function call and return parameter
        name completions (e.g., dtype=, order=, copy=).

        Returns (items, call_info) where call_info is None if not in a call.
        """
        call_info = self._find_enclosing_call(code, cursor_pos)
        if not call_info:
            return [], None

        func_name = call_info["func_name"]
        existing_kwargs = call_info["existing_kwargs"]
        prefix = call_info["prefix"]

        # Resolve the function object and get its parameters
        try:
            obj = eval(func_name, {"__builtins__": __builtins__}, self.shell.user_ns)
        except Exception:
            return [], call_info

        params = self._extract_params(obj)
        if not params:
            return [], call_info

        items = []
        for entry in params:
            name = entry[0]
            detail = entry[1] if len(entry) > 1 else None
            doc = entry[2] if len(entry) > 2 else None

            if name in existing_kwargs:
                continue

            kw_label = f"{name}="
            if prefix and not kw_label.startswith(prefix):
                continue

            items.append(
                CompletionItem(
                    label=kw_label,
                    insertText=kw_label,
                    kind="field",
                    detail=detail or name,
                    documentation=doc,
                    sortText=f"0{name}",  # sort before everything
                )
            )

        return items, call_info

    @staticmethod
    def _extract_params(obj) -> list[tuple[str, str | None]]:
        """Extract parameter names and detail strings from a callable.

        Tries inspect.signature() first, falls back to parsing the
        docstring signature for C builtins like np.array.

        Returns list of (name, detail) tuples. Skips *args/**kwargs.
        """
        import inspect
        import re

        # Strategy 1: inspect.signature (works for pure Python functions)
        try:
            sig = inspect.signature(obj)
            params = []
            has_only_var = True  # Track if signature is only *args/**kwargs
            for name, p in sig.parameters.items():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                has_only_var = False
                detail = name
                ann = ""
                if p.annotation is not p.empty:
                    try:
                        ann = (
                            p.annotation.__name__
                            if hasattr(p.annotation, "__name__")
                            else str(p.annotation)
                        )
                    except Exception:
                        pass
                if p.default is not p.empty:
                    default_repr = repr(p.default)
                    if len(default_repr) > 50:
                        default_repr = default_repr[:50] + "…"
                    detail = f"{name}: {ann} = {default_repr}" if ann else f"{name}={default_repr}"
                elif ann:
                    detail = f"{name}: {ann}"
                params.append((name, detail))
            # If we got real params (not just *args/**kwargs), use them.
            # Enrich with per-param docs from docstring.
            if params and not has_only_var:
                doc = inspect.getdoc(obj)
                if doc:
                    doc_params = IPythonWorker._parse_docstring_params(doc)
                    doc_map = {dp[0]: dp[2] for dp in doc_params if len(dp) > 2}
                    params = [(n, d, doc_map.get(n)) for n, d in params]
                else:
                    params = [(n, d, None) for n, d in params]
                return params
        except (ValueError, TypeError):
            pass

        # Strategy 2: Parse Args/Parameters section from docstring.
        # Handles **kwargs functions where the docstring documents the
        # actual accepted keyword arguments.
        doc = inspect.getdoc(obj)
        if doc:
            doc_params = IPythonWorker._parse_docstring_params(doc)
            if doc_params:
                return doc_params

        # Strategy 3: Parse the docstring signature for C builtins.
        # e.g., "array(object, dtype=None, *, copy=True, order='K', ...)"
        # The signature may span multiple lines (numpy style).
        doc = inspect.getdoc(obj)
        if not doc:
            return []

        # Join continuation lines: collect lines until we find the closing ')'
        lines = doc.split("\n")
        sig_text = ""
        for line in lines:
            sig_text += " " + line.strip()
            if ")" in sig_text:
                break

        sig_text = sig_text.strip()
        # Match "funcname(params...)"
        m = re.match(r'[\w.]+\((.+)\)', sig_text)
        if not m:
            return []

        arg_str = m.group(1)

        # Also parse the Parameters section for per-param docs
        doc_params = IPythonWorker._parse_docstring_params(doc)
        doc_map = {dp[0]: dp for dp in doc_params}

        params = []
        # Split on commas, respecting brackets and parens
        depth = 0
        current = ""
        for ch in arg_str + ",":
            if ch in "([{":
                depth += 1
                current += ch
            elif ch in ")]}":
                depth -= 1
                current += ch
            elif ch == "," and depth == 0:
                token = current.strip()
                current = ""
                if not token or token == "*" or token == "/":
                    continue
                if token.startswith("*") or token.startswith("**"):
                    continue
                # Parse "name=default" or "name"
                eq = token.find("=")
                if eq >= 0:
                    name = token[:eq].strip().rstrip(": ")
                    colon = name.find(":")
                    if colon >= 0:
                        name = name[:colon].strip()
                else:
                    colon = token.find(":")
                    if colon >= 0:
                        name = token[:colon].strip()
                    else:
                        name = token.strip()

                # Get per-param doc from Parameters section
                dp = doc_map.get(name)
                detail = dp[1] if dp and len(dp) > 1 and dp[1] else token
                pdoc = dp[2] if dp and len(dp) > 2 else None
                params.append((name, detail, pdoc))
            else:
                current += ch

        return params

    def _find_enclosing_call(self, code: str, cursor_pos: int) -> dict | None:
        """Find the function being called at cursor position.

        Walks backward from cursor to find the matching unmatched '(',
        then extracts the function name.

        Returns {func_name, existing_kwargs, prefix} or None.
        """
        text = code[:cursor_pos]

        # Find the unmatched open paren
        depth = 0
        paren_pos = -1
        i = len(text) - 1
        while i >= 0:
            ch = text[i]
            if ch == ')':
                depth += 1
            elif ch == '(':
                if depth == 0:
                    paren_pos = i
                    break
                depth -= 1
            i -= 1

        if paren_pos < 0:
            return None

        # Extract the function name before the '('
        # Walk backward from paren to get e.g. "np.array", "dspy.configure"
        func_end = paren_pos
        j = paren_pos - 1
        while j >= 0 and (text[j].isalnum() or text[j] in '._'):
            j -= 1
        func_name = text[j + 1 : func_end].strip()
        if not func_name:
            return None

        # Extract text inside the parens (to find already-used kwargs)
        inside = text[paren_pos + 1 :]
        existing_kwargs = set()
        # Simple regex to find "name=" patterns (not inside strings)
        import re
        for m in re.finditer(r'(?<![=!<>])(\w+)\s*=(?!=)', inside):
            existing_kwargs.add(m.group(1))

        # Extract the current prefix (what user is typing for this arg)
        # Find text after last comma or after '('
        after_last_sep = inside
        for sep_pos in range(len(inside) - 1, -1, -1):
            if inside[sep_pos] in ',(':
                after_last_sep = inside[sep_pos + 1 :]
                break
        prefix = after_last_sep.strip()

        return {
            "func_name": func_name,
            "existing_kwargs": existing_kwargs,
            "prefix": prefix,
        }

    @staticmethod
    def _parse_docstring_params(doc: str) -> list[tuple[str, str | None, str | None]]:
        """Parse parameter names and their full descriptions from a docstring.

        Returns list of (name, detail, documentation) tuples.

        Supports google-style:
            Args:
                name (type): description
                    continued description

        And numpy-style:
            Parameters
            ----------
            name : type
                description
                continued description
        """
        import re

        lines = doc.split("\n")
        params: list[tuple[str, str | None, str | None]] = []
        in_section = False
        section_indent = 0
        # Track current param being collected
        current_name: str | None = None
        current_detail: str | None = None
        current_doc_lines: list[str] = []
        param_indent: int | None = None  # indent of param definition lines

        def flush_param():
            nonlocal current_name, current_detail, current_doc_lines
            if current_name:
                doc_text = "\n".join(current_doc_lines).strip() or None
                params.append((current_name, current_detail, doc_text))
            current_name = None
            current_detail = None
            current_doc_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()
            line_indent = len(line) - len(line.lstrip()) if stripped else 999

            # Detect section headers
            if re.match(r'^(Args|Arguments|Parameters|Params|Keyword Args|Keyword Arguments):?\s*$', stripped, re.IGNORECASE):
                flush_param()
                in_section = True
                section_indent = line_indent
                param_indent = None
                # numpy-style: skip the "----" underline
                if i + 1 < len(lines) and re.match(r'^-+\s*$', lines[i + 1].strip()):
                    continue
                continue

            if not in_section:
                continue

            # Detect end of section (new section header or dedent)
            if stripped and line_indent <= section_indent and not stripped.startswith('-'):
                if re.match(r'^[A-Z][\w\s]+:?\s*$', stripped):
                    flush_param()
                    in_section = False
                    continue

            # Empty line inside section
            if not stripped:
                if current_name:
                    current_doc_lines.append("")
                continue

            # Skip *args/**kwargs entries themselves
            if stripped.startswith('**') or stripped.startswith('*'):
                continue

            # Try to match a new parameter definition
            # Google-style: "name (type): description" or "name: description"
            m = re.match(r'^([a-z_]\w*)\s*(?:\(([^)]*)\))?\s*:\s*(.*)', stripped)
            if not m:
                # Numpy-style: "name : type"
                m = re.match(r'^([a-z_]\w*)\s*:\s*(.+)', stripped)
                if m:
                    # Numpy-style matched — type is group 2
                    flush_param()
                    current_name = m.group(1)
                    ptype = m.group(2).strip()
                    current_detail = f"{current_name}: {ptype}"
                    param_indent = line_indent
                    continue

            if m and m.lastindex and m.lastindex >= 1:
                # Check if this is at the param indent level (not a continuation)
                if param_indent is not None and line_indent > param_indent + 2:
                    # This is a continuation line, not a new param
                    if current_name:
                        current_doc_lines.append(stripped)
                    continue

                flush_param()
                current_name = m.group(1)
                ptype = m.group(2) if m.lastindex >= 2 and m.group(2) else ""
                desc = m.group(m.lastindex).strip() if m.lastindex >= 2 else (m.group(2) or "").strip()

                # For google-style with 3 groups: name, type, desc
                if m.lastindex >= 3:
                    ptype = m.group(2) or ""
                    desc = m.group(3).strip()

                if ptype:
                    current_detail = f"{current_name}: {ptype}"
                else:
                    current_detail = current_name
                if desc:
                    current_doc_lines.append(desc)
                if param_indent is None:
                    param_indent = line_indent
                continue

            # Continuation line for the current param's description
            if current_name and line_indent > (param_indent or 0):
                current_doc_lines.append(stripped)

        flush_param()
        return params

    @staticmethod
    def _looks_like_path(text: str) -> bool:
        """Check if a completion looks like a filesystem path."""
        # Contains path separators or file extensions
        if '/' in text or '\\' in text:
            return True
        # Looks like a filename with extension (e.g., AGENTS.md, setup.py)
        if '.' in text and not text.startswith('.') and not text.startswith('__'):
            parts = text.rsplit('.', 1)
            if len(parts) == 2 and parts[1].isalpha() and len(parts[1]) <= 5:
                # Check it's not a valid Python dotted name in the namespace
                return True
        return False

    def _completion_detail(
        self, label: str, comp_type: str | None, code: str, cursor_pos: int
    ) -> tuple[str | None, str | None, str]:
        """Get detail, documentation, and sort prefix for a completion item.

        Returns (detail, documentation, sortPrefix).
        Keeps it fast — skips expensive lookups for magics/keywords.
        """
        # Sort prefix: push IPython magics and dunders to bottom
        if label.startswith("%%") or label.startswith("%"):
            return ("IPython magic", None, "z")  # sort last
        if label.startswith("__") and label.endswith("__"):
            return (comp_type, None, "y")  # sort near bottom
        if label.startswith("_"):
            return (comp_type, None, "x")  # sort below public

        # For real symbols, try to get signature + short docstring.
        # Build the fully qualified name from the code before the cursor.
        # Find the last '.' before cursor to get the base object.
        prefix = code[:cursor_pos]
        dot_idx = prefix.rfind(".")
        if dot_idx >= 0:
            # "dspy.co" → base = "dspy", fqn = "dspy.configure"
            base = prefix[:dot_idx].rstrip()
            fqn = f"{base}.{label}"
        else:
            fqn = label

        try:
            obj = eval(fqn, {"__builtins__": {}}, self.shell.user_ns)

            # Detail: type + signature
            detail = type(obj).__name__
            if callable(obj):
                try:
                    import inspect
                    sig = inspect.signature(obj)
                    detail = f"{label}{sig}"
                except (ValueError, TypeError):
                    detail = f"{label}(...)"

            # Documentation: first paragraph of docstring
            doc = None
            try:
                import inspect
                raw_doc = inspect.getdoc(obj)
                if raw_doc:
                    doc = raw_doc
            except Exception:
                pass

            return (detail, doc, "a")  # "a" = sort first (real matches)
        except Exception:
            return (comp_type, None, "b")  # sort normally

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

            # detail_level=1 tells IPython to include source code
            ipython_detail = 1 if detail >= 2 else 0
            info = self.shell.object_inspect(name, detail_level=ipython_detail)
            if not info.get("found"):
                return InspectResult(found=False, name=name)

            # If IPython didn't get source, try inspect.getsource() directly
            source_code = info.get("source")
            if detail >= 2 and not source_code:
                try:
                    import inspect
                    obj = eval(name, {"__builtins__": {}}, self.shell.user_ns)
                    source_code = inspect.getsource(obj)
                except Exception:
                    pass

            file_path = info.get("file")
            line_num = info.get("line")

            # IPython often returns file but not line. Use inspect to get both.
            if not line_num or not file_path:
                try:
                    import inspect as _inspect
                    obj = eval(name, {"__builtins__": {}}, self.shell.user_ns)
                    # Unwrap decorated functions to get the real source
                    obj = _inspect.unwrap(obj, stop=(lambda f: hasattr(f, '__code__')))
                    src_file = _inspect.getfile(obj)
                    _, src_line = _inspect.getsourcelines(obj)
                    file_path = file_path or src_file
                    line_num = line_num or src_line
                except Exception:
                    pass

            # Expand ~ in file paths
            if file_path and file_path.startswith("~"):
                import os
                file_path = os.path.expanduser(file_path)

            return InspectResult(
                found=True,
                source="runtime",
                name=name,
                kind=self._classify_object_kind(info),
                type=info.get("type_name"),
                signature=info.get("call_signature") or info.get("init_signature"),
                docstring=info.get("docstring") if detail >= 1 else None,
                sourceCode=source_code if detail >= 2 else None,
                file=file_path,
                line=line_num,
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

                docstring = None
                try:
                    import inspect
                    docstring = inspect.getdoc(value)
                except Exception:
                    docstring = None

                return HoverResult(
                    found=True,
                    name=name,
                    type=type_name,
                    value=value_str,
                    signature=signature,
                    docstring=docstring,
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
                    docstring=info.get("docstring"),
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

    def get_history(
        self,
        n: int = 20,
        pattern: str | None = None,
        before: int | None = None,
    ) -> HistoryResult:
        """Get persistent IPython history."""
        if self._should_use_subprocess():
            return self._get_subprocess_worker().get_history(n=n, pattern=pattern, before=before)

        self._ensure_initialized()

        try:
            if n <= 0:
                n = 20

            history_manager = self.shell.history_manager
            search_pattern = pattern or "*"
            rows = list(history_manager.search(search_pattern, raw=True, output=False, unique=False))

            entries: list[HistoryEntry] = []
            for row in rows:
                if len(row) < 3:
                    continue
                session_id = int(row[0])
                line_number = int(row[1])
                code = row[2] or ""
                history_index = self._make_history_index(session_id, line_number)
                if before is not None and history_index >= before:
                    continue
                entries.append(HistoryEntry(historyIndex=history_index, code=code))

            start = max(0, len(entries) - n)
            return HistoryResult(entries=entries[start:], hasMore=start > 0)
        except Exception:
            return HistoryResult(entries=[], hasMore=False)

    def _make_history_index(self, session_id: int, line_number: int) -> int:
        """Encode IPython's (session, line) history key as a stable integer."""
        return (int(session_id) << 32) | int(line_number)

    # =========================================================================
    # Session Management
    # =========================================================================

    def reset(self):
        """Reset the namespace."""
        if self._subprocess_worker is not None:
            self._subprocess_worker.reset()
        elif self._initialized:
            self.shell.reset()

    def clear_history(self) -> bool:
        """Clear execution history if supported."""
        try:
            if self._subprocess_worker is not None:
                return self._subprocess_worker.clear_history()
            self._ensure_initialized()
            self.shell.history_manager.reset()
            return True
        except Exception:
            return False

    def shutdown(self):
        """Shutdown the worker, cleaning up subprocess if needed."""
        if self._subprocess_worker is not None:
            self._subprocess_worker.shutdown()
            self._subprocess_worker = None

    def interrupt(self) -> bool:
        """Interrupt currently running code.

        For subprocess workers, sends SIGINT.
        For local execution, raises KeyboardInterrupt in the executing thread.

        Returns True if interrupt was initiated.
        """
        if self._subprocess_worker is not None:
            return self._subprocess_worker.interrupt()

        # For local execution, raise KeyboardInterrupt in the executing thread
        thread_id = self._executing_thread_id
        if thread_id is not None:
            import ctypes
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(thread_id),
                ctypes.py_object(KeyboardInterrupt),
            )
            if res == 1:
                return True
            elif res > 1:
                # "If it returns a number greater than one, you're in trouble,
                # and you should call it again with exc=NULL to revert the effect"
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(thread_id), None
                )
            return False

        return False

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
