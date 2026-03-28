"""
MRP MCP Server — the primary interface for mrmd-python.

3 tools:
- py:       run code or provide input
- py_look:  see runtime state, variables, or inspect a symbol
- py_ctl:   reset or cancel

That's it.
"""

import json
import re
from contextlib import asynccontextmanager

from fastmcp import FastMCP

from .service import RuntimeService


# =============================================================================
# Formatters — turn verbose dicts into compact text the LLM actually needs
# =============================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _clean(text: str | None) -> str:
    if not text:
        return ""
    s = _strip_ansi(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    # Strip IPython's "Out[N]: " prefix from expression results
    s = re.sub(r"^Out\[\d+\]: ", "", s)
    return s


def _truncate(text: str, max_lines: int = 100) -> str:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    keep = max_lines // 2
    omitted = len(lines) - max_lines
    return "\n".join(
        lines[:keep]
        + [f"… [{omitted} lines truncated] …"]
        + lines[-keep:]
    )


def _clip(text: str, max_chars: int = 200) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15] + " … [truncated]"


def _dedup(*blocks: str) -> str:
    """Join non-empty blocks, skipping duplicates."""
    seen = set()
    parts = []
    for b in blocks:
        b = b.strip()
        if b and b not in seen:
            seen.add(b)
            parts.append(b)
    return "\n".join(parts)


def _state_hint(result: dict) -> str:
    """Build the ✓/✗ state hint line."""
    success = result.get("success", True)
    marker = "✓" if success else "✗"
    parts = []

    duration = result.get("durationMs") or result.get("duration")
    if duration is not None:
        if duration < 1000:
            parts.append(f"{duration}ms")
        else:
            parts.append(f"{duration / 1000:.1f}s")

    var_count = result.get("_varCount")
    if var_count is not None:
        parts.append(f"{var_count} var{'s' if var_count != 1 else ''}")

    assets = result.get("assets", [])
    if assets:
        n = len(assets)
        parts.append(f"{n} asset{'s' if n != 1 else ''}")

    return f"{marker} {' | '.join(parts)}" if parts else marker


def format_run(result: dict) -> str:
    """Format an execution result for the LLM."""
    stdout = _clean(result.get("stdout"))
    expr_result = _clean(result.get("result"))
    stderr = _clean(result.get("stderr"))
    error = result.get("error")

    lines = []

    if error:
        # On error: show condensed error, skip raw traceback from stdout
        if isinstance(error, dict):
            err_type = error.get("type", "Error")
            err_msg = error.get("message", "")
            lines.append(f"{err_type}: {err_msg}")
            err_line = error.get("line")
            tb = error.get("traceback", [])
            if err_line is not None:
                # Find the offending code line in traceback
                code_line = None
                for t in reversed(tb):
                    t = _strip_ansi(t).strip()
                    if (t and not t.startswith("Traceback") and
                            not t.startswith("File ") and
                            not t.startswith("Cell ") and
                            "Error" not in t and
                            "--->" not in t and
                            t != ""):
                        code_line = t
                        break
                if code_line:
                    lines.append(f"  line {err_line}: {code_line}")
                else:
                    lines.append(f"  line {err_line}")
        else:
            lines.append(str(error))
    else:
        # On success: show output
        main = _dedup(stdout, expr_result)
        if main:
            lines.append(_truncate(main))

        # Stderr (only when no error)
        if stderr:
            if lines:
                lines.append("")
            lines.append(f"STDERR: {_clip(stderr, 240)}")

    # State hint
    lines.append("")
    lines.append(_state_hint(result))

    return "\n".join(lines)


def format_input_requested(state: dict) -> str:
    """Format an input_required response."""
    ir = state.get("inputRequest", {})
    prompt = ir.get("prompt", "")
    lines = [
        f'INPUT REQUESTED: "{prompt}"',
        "",
        "⏳ use py(input=\"...\") to continue",
    ]
    return "\n".join(lines)


def format_overview(health: dict, variables: dict) -> str:
    """Format the default py_look() response."""
    state = health.get("state", "?")
    var_list = variables.get("variables", [])
    exec_count = health.get("executionCount", 0)

    lines = [f"python {state} | {len(var_list)} vars | exec #{exec_count}"]

    if var_list:
        lines.append("")
        # Find column widths
        name_w = max(len(v.get("name", "")) for v in var_list[:25])
        type_w = max(len(v.get("type", "")) for v in var_list[:25])
        name_w = max(name_w, 4)
        type_w = max(type_w, 4)

        for v in var_list[:25]:
            name = v.get("name", "")
            vtype = v.get("type", "")
            value = _clip(v.get("value", ""), 80)
            lines.append(f"{name:<{name_w}}  {vtype:<{type_w}}  {value}")

        if len(var_list) > 25:
            lines.append(f"… {len(var_list) - 25} more")
    else:
        lines.append("")
        lines.append("No variables.")

    return "\n".join(lines)


def format_inspect(result: dict, symbol: str) -> str:
    """Format a py_look(at=...) response."""
    if not result.get("found"):
        return f"{symbol}: not found in runtime namespace"

    lines = []
    name = result.get("name") or symbol
    kind = result.get("kind") or ""
    vtype = result.get("type") or ""

    header = name
    if vtype:
        header += f": {vtype}"
    if kind and kind != vtype:
        header += f" ({kind})"
    lines.append(header)

    sig = result.get("signature")
    if sig:
        lines.append(f"  {sig}")

    value = result.get("value")
    if value:
        lines.append(f"  Value: {_clip(value, 300)}")

    # Shape/size info from variable detail
    shape = result.get("shape")
    if shape:
        lines.append(f"  Shape: {shape}")

    length = result.get("length")
    if length is not None:
        lines.append(f"  Length: {length}")

    children = result.get("children")
    if isinstance(children, int) and children > 0:
        lines.append(f"  Children: {children}")

    loc = result.get("file")
    if loc:
        line_num = result.get("line")
        lines.append(f"  Defined in: {loc}{f':{line_num}' if line_num else ''}")

    doc = result.get("docstring")
    if doc:
        lines.append("")
        lines.append(_clip(doc, 500))

    return "\n".join(lines)


def format_reset(result: dict) -> str:
    cleared = ", ".join(result.get("cleared", []))
    var_count = 0  # after reset, always 0
    return f"RESET | {cleared} cleared | {var_count} vars"


def format_cancel(result: dict) -> str:
    exec_id = result.get("execution", {}).get("executionId", "?")
    return f"CANCELLED {exec_id}"


# =============================================================================
# MCP Server
# =============================================================================

INSTRUCTIONS = (
    "Python runtime. "
    "Use py to run code, py_look to see variables and inspect symbols, "
    "py_ctl to reset or cancel."
)


def create_mcp_server(
    service: RuntimeService | None = None,
    name: str = "mrmd-python",
    **service_kwargs,
) -> FastMCP:
    """Create the MCP server with 3 agent-friendly tools."""

    if service is None:
        service = RuntimeService(**service_kwargs)

    @asynccontextmanager
    async def lifespan(server):
        yield
        service.shutdown()

    mcp = FastMCP(name=name, instructions=INSTRUCTIONS, lifespan=lifespan)

    # ── Resources ────────────────────────────────────────────

    @mcp.resource("mrp://health")
    def health_resource() -> str:
        """Runtime health and current state."""
        return json.dumps(service.get_health())

    @mcp.resource("mrp://capabilities")
    def capabilities_resource() -> str:
        """Runtime capabilities and feature flags."""
        return json.dumps(service.get_capabilities())

    # ── py: run code or provide input ────────────────────────

    @mcp.tool
    def py(
        code: str | None = None,
        input: str | None = None,
        allow_input: bool = True,
    ) -> str:
        """Run Python code or provide input to a waiting execution.

        code: Python code to execute.
        input: Text to send to a waiting input() prompt.
        allow_input: If false, input() calls fail instead of waiting (default true).
        """
        if code and input:
            return "ERROR: provide either code or input, not both"

        if input is not None:
            # Provide input to waiting execution
            try:
                status, data = service.provide_input_and_wait(input)
            except Exception as e:
                return f"ERROR: {e}"

            if status == "input_required":
                return format_input_requested(data)
            else:
                # Terminal — enrich with var count
                _enrich_var_count(service, data)
                return format_run(data)

        if not code:
            return "ERROR: provide code to execute"

        # Execute code
        try:
            status, data = service.execute_until_input_or_done(
                code=code,
                allow_input=allow_input,
            )
        except RuntimeError as e:
            return f"ERROR: {e}"

        if status == "input_required":
            return format_input_requested(data)
        else:
            _enrich_var_count(service, data)
            return format_run(data)

    # ── py_look: see state, variables, or inspect ────────────

    @mcp.tool
    def py_look(at: str | None = None) -> str:
        """See runtime state, variables, or inspect a symbol.

        No args: overview with variable list.
        at="df": inspect the symbol df.
        at="df.columns": drill into nested attribute.
        """
        if at is None:
            # Overview: health + variable list
            health = service.get_health()
            try:
                variables = service.list_variables()
            except RuntimeError:
                variables = {"variables": [], "count": 0, "truncated": False, "warnings": []}
            return format_overview(health, variables)

        # Inspect a symbol — try inspect first, fall back to hover
        try:
            result = service.inspect(code=at, cursor=len(at), detail=1)
            if result.get("found"):
                return format_inspect(result, at)
        except RuntimeError:
            pass

        # Try hover
        try:
            result = service.hover(code=at, cursor=len(at))
            if result.get("found"):
                return format_inspect(result, at)
        except RuntimeError:
            pass

        # Try variable detail if it looks like a name
        if re.match(r"^[a-zA-Z_]\w*$", at):
            try:
                detail = service.get_variable(name=at)
                if detail.get("name"):
                    return format_inspect(detail, at)
            except (RuntimeError, KeyError):
                pass

        return f"{at}: not found in runtime namespace"

    # ── py_ctl: control ──────────────────────────────────────

    @mcp.tool
    def py_ctl(op: str, scope: str = "namespace") -> str:
        """Control the runtime.

        op="reset": clear namespace (scope: "namespace", "history", "assets", or "all").
        op="cancel": cancel the active execution.
        """
        if op == "reset":
            scopes = ["namespace", "history", "assets"] if scope == "all" else [scope]
            try:
                result = service.reset(scope=scopes)
                return format_reset(result)
            except RuntimeError as e:
                return f"ERROR: {e}"

        elif op == "cancel":
            health = service.get_health()
            exec_id = health.get("currentExecutionId")
            if not exec_id:
                return "No active execution."
            try:
                result = service.cancel_execution(exec_id)
                return format_cancel(result)
            except RuntimeError as e:
                return f"ERROR: {e}"

        return f"ERROR: unknown op '{op}'. Use 'reset' or 'cancel'."

    return mcp


def _enrich_var_count(service: RuntimeService, result: dict):
    """Try to add variable count to result for the state hint."""
    try:
        variables = service.list_variables()
        result["_varCount"] = variables.get("count", len(variables.get("variables", [])))
    except Exception:
        pass
