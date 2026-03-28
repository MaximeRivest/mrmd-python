"""
MCP Server — the primary interface for rat-py (Run AnyThing: Python).

3 tools:
- py:       run code or provide input (streams stdout/stderr in real-time)
- py_look:  see state, variables, inspect a symbol, or get completions
- py_ctl:   reset or cancel

Same tools serve both AI agents and notebook GUIs.
- Agents read the compact text result.
- GUIs listen to real-time logging notifications (stdout, stderr, display, asset).
- GUIs call py_look(code=..., cursor=N) for completions.
"""

import asyncio
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP, Context
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse

from .service import RuntimeService


# =============================================================================
# Formatters
# =============================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _clean(s: str | None) -> str:
    if not s:
        return ""
    s = _strip_ansi(s).replace("\r\n", "\n").replace("\r", "\n").strip()
    return re.sub(r"^Out\[\d+\]: ", "", s)


def _truncate(s: str, n: int = 100) -> str:
    lines = s.split("\n")
    if len(lines) <= n:
        return s
    k = n // 2
    return "\n".join(lines[:k] + [f"… [{len(lines) - n} lines truncated] …"] + lines[-k:])


def _clip(s: str, n: int = 200) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 15] + " … [truncated]"


def _dedup(*blocks: str) -> str:
    seen = set()
    out = []
    for b in blocks:
        b = b.strip()
        if b and b not in seen:
            seen.add(b)
            out.append(b)
    return "\n".join(out)


def _hint(result: dict) -> str:
    ok = result.get("success", True)
    parts = []
    ms = result.get("durationMs") or result.get("duration")
    if ms is not None:
        parts.append(f"{ms}ms" if ms < 1000 else f"{ms / 1000:.1f}s")
    vc = result.get("_varCount")
    if vc is not None:
        parts.append(f"{vc} var{'s' if vc != 1 else ''}")
    na = len(result.get("assets", []))
    if na:
        parts.append(f"{na} asset{'s' if na != 1 else ''}")
    return ("✓ " if ok else "✗ ") + " | ".join(parts) if parts else ("✓" if ok else "✗")


def _fmt_run(r: dict) -> str:
    stdout = _clean(r.get("stdout"))
    expr = _clean(r.get("result"))
    stderr = _clean(r.get("stderr"))
    error = r.get("error")
    lines = []

    if error:
        if isinstance(error, dict):
            lines.append(f"{error.get('type', 'Error')}: {error.get('message', '')}")
            ln = error.get("line")
            if ln is not None:
                # Find offending code in traceback
                for t in reversed(error.get("traceback", [])):
                    t = _strip_ansi(t).strip()
                    if (t and "Traceback" not in t and "File " not in t and
                            "Cell " not in t and "Error" not in t and "--->" not in t):
                        lines.append(f"  line {ln}: {t}")
                        break
                else:
                    lines.append(f"  line {ln}")
        else:
            lines.append(str(error))
    else:
        main = _dedup(stdout, expr)
        if main:
            lines.append(_truncate(main))
        if stderr:
            lines.append(f"\nSTDERR: {_clip(stderr, 240)}" if lines else f"STDERR: {_clip(stderr, 240)}")

    for a in r.get("assets", []):
        lines.append(f"[{a.get('assetType', 'file')}: {a.get('uri', '')}]")

    lines.append("")
    lines.append(_hint(r))
    return "\n".join(lines)


def _fmt_input(state: dict) -> str:
    ir = state.get("inputRequest") or {}
    return f'INPUT REQUESTED: "{ir.get("prompt", "")}"\n\n⏳ use py(input="...") to continue'


def _fmt_overview(health: dict, variables: dict) -> str:
    vl = variables.get("variables", [])
    lines = [f"python {health.get('state', '?')} | {len(vl)} vars | exec #{health.get('executionCount', 0)}"]
    if vl:
        lines.append("")
        nw = max((len(v.get("name", "")) for v in vl[:25]), default=4)
        tw = max((len(v.get("type", "")) for v in vl[:25]), default=4)
        for v in vl[:25]:
            lines.append(f"{v['name']:<{max(nw,4)}}  {v.get('type',''):<{max(tw,4)}}  {_clip(v.get('value',''), 80)}")
        if len(vl) > 25:
            lines.append(f"… {len(vl) - 25} more")
    else:
        lines += ["", "No variables."]
    return "\n".join(lines)


def _fmt_inspect(r: dict, sym: str) -> str:
    if not r.get("found"):
        return f"{sym}: not found in runtime namespace"
    lines = []
    name = r.get("name") or sym
    kind = r.get("kind") or ""
    vtype = r.get("type") or ""
    value = r.get("value") or ""

    # Header: name: type (size)
    h = name
    if vtype:
        h += f": {vtype}"
    if kind and kind not in ("variable", vtype):
        h += f" ({kind})"
    size = r.get("size")
    if size:
        h += f" ({size})"
    elif r.get("shape"):
        h += f" {r['shape']}"
    elif r.get("length") is not None:
        h += f" ({r['length']} items)"
    lines.append(h)

    # Value
    if kind == "variable" and value:
        lines.append(f"  = {_clip(value, 300)}")
    elif r.get("signature"):
        lines.append(f"  {r['signature']}")
    elif kind != "variable" and value:
        lines.append(f"  Value: {_clip(value, 200)}")

    # Location
    if r.get("file"):
        loc = r["file"]
        if r.get("line"):
            loc += f":{r['line']}"
        lines.append(f"  Defined in: {loc}")

    # Docstring (for functions/methods/classes, not trivial variables)
    if r.get("docstring") and kind in ("function", "method", "class", "module", ""):
        doc = r["docstring"]
        if kind != "variable" or len(doc) < 200:
            lines.append("")
            lines.append(_clip(doc, 500))

    # Children — the drill-down view
    children = r.get("children")
    if children:
        lines.append("")
        nw = max((len(str(c.get("name", ""))) for c in children[:20]), default=4)
        tw = max((len(str(c.get("type", ""))) for c in children[:20]), default=4)
        for c in children[:20]:
            cname = str(c.get("name", ""))
            ctype = str(c.get("type", ""))
            cval = _clip(str(c.get("value", "")), 60)
            exp = "▸" if c.get("expandable") else " "
            lines.append(f"  {exp} {cname:<{max(nw,4)}}  {ctype:<{max(tw,4)}}  {cval}")
        remaining = len(children) - 20
        if remaining > 0:
            lines.append(f"  … {remaining} more")

    # Methods and attributes hints
    methods = r.get("methods")
    if methods:
        lines.append("")
        lines.append(f"  Methods: {', '.join(methods[:10])}")
        if len(methods) > 10:
            lines.append(f"  … {len(methods) - 10} more")

    attrs = r.get("attributes")
    if attrs:
        lines.append(f"  Attributes: {', '.join(attrs[:10])}")
        if len(attrs) > 10:
            lines.append(f"  … {len(attrs) - 10} more")

    return "\n".join(lines)


def _fmt_completions(r: dict) -> str:
    matches = r.get("matches", [])
    if not matches:
        return "No completions."
    lw = max(len(m.get("label", "")) for m in matches[:20])
    kw = max(len(m.get("kind", "")) for m in matches[:20])
    lines = []
    for m in matches[:20]:
        lines.append(f"{m['label']:<{max(lw,4)}}  {m.get('kind',''):<{max(kw,4)}}  {m.get('detail') or m.get('type') or ''}")
    if len(matches) > 20:
        lines.append(f"… {len(matches) - 20} more")
    return "\n".join(lines)


# =============================================================================
# Streaming helpers
# =============================================================================

async def _emit_event(ctx: Context, event):
    """Send an execution event as an MCP logging notification."""
    etype = event.type if hasattr(event, "type") else event.get("type")
    content = event.content if hasattr(event, "content") else event.get("content")

    if etype == "stdout" and content:
        await ctx.info(_strip_ansi(content), logger_name="stdout")
    elif etype == "stderr" and content:
        await ctx.warning(_strip_ansi(content), logger_name="stderr")
    elif etype == "display":
        dd = event.displayData if hasattr(event, "displayData") else event.get("displayData")
        if dd:
            data = dd.data if hasattr(dd, "data") else dd.get("data", {})
            mimes = list(data.keys()) if isinstance(data, dict) else []
            await ctx.info(json.dumps({"display": mimes, "data": data}), logger_name="display")
    elif etype == "asset":
        asset = event.asset if hasattr(event, "asset") else event.get("asset")
        if asset:
            uri = asset.uri if hasattr(asset, "uri") else asset.get("uri", "")
            atype = asset.assetType if hasattr(asset, "assetType") else asset.get("assetType", "file")
            await ctx.info(f"[{atype}: {uri}]", logger_name="asset")
    elif etype == "input_requested":
        ir = event.inputRequest if hasattr(event, "inputRequest") else event.get("inputRequest")
        if ir:
            prompt = ir.prompt if hasattr(ir, "prompt") else ir.get("prompt", "")
            await ctx.info(json.dumps({"input_requested": prompt}), logger_name="input")
    elif etype == "warning":
        w = event.warning if hasattr(event, "warning") else event.get("warning")
        if w:
            msg = w.message if hasattr(w, "message") else w.get("message", "")
            await ctx.warning(msg, logger_name="runtime")


async def _stream_until_done(service, execution, ctx):
    """Poll execution, stream events, return when terminal or input_required."""
    last_seq = 0
    while not execution.is_terminal and execution.status != "input_required":
        await asyncio.sleep(0.05)
        new = execution.events[last_seq:]
        for ev in new:
            if ctx:
                await _emit_event(ctx, ev)
        last_seq = len(execution.events)

    # Drain
    for ev in execution.events[last_seq:]:
        if ctx:
            await _emit_event(ctx, ev)


# =============================================================================
# MCP Server
# =============================================================================

INSTRUCTIONS = (
    "Python runtime. "
    "run(code) runs code. "
    "look() shows variables. look(at='x') inspects x. "
    "look(code='df.he', cursor=5) completes. "
    "ctl(op='reset') resets."
)


def create_mcp_server(
    service: RuntimeService | None = None,
    name: str = "rat-py",
    **service_kwargs,
) -> FastMCP:
    if service is None:
        service = RuntimeService(**service_kwargs)

    @asynccontextmanager
    async def lifespan(server):
        yield
        service.shutdown()

    mcp = FastMCP(name=name, instructions=INSTRUCTIONS, lifespan=lifespan)

    # ── Custom HTTP routes (served alongside /mcp) ───────────

    @mcp.custom_route("/health", methods=["GET"])
    async def health_http(request: Request) -> JSONResponse:
        return JSONResponse(service.get_health())

    @mcp.custom_route("/assets/{asset_id}", methods=["GET"])
    async def asset_http(request: Request) -> FileResponse | JSONResponse:
        asset_id = request.path_params["asset_id"]
        asset_path = service.get_asset_path(asset_id)
        if asset_path is None:
            return JSONResponse(
                {"error": "asset_not_found", "message": f"Asset {asset_id} not found"},
                status_code=404,
            )
        suffix = asset_path.suffix.lower()
        content_types = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml", ".html": "text/html", ".json": "application/json",
        }
        return FileResponse(asset_path, media_type=content_types.get(suffix, "application/octet-stream"))

    # ── Resources ────────────────────────────────────────────

    @mcp.resource("mrp://health")
    def health_resource() -> str:
        return json.dumps(service.get_health())

    @mcp.resource("mrp://capabilities")
    def capabilities_resource() -> str:
        return json.dumps(service.get_capabilities())

    # ── py ───────────────────────────────────────────────────

    @mcp.tool
    async def run(
        code: str | None = None,
        input: str | None = None,
        allow_input: bool = True,
        ctx: Context = None,
    ) -> str:
        """Run Python code or provide input to a waiting execution.

        code: Python code to execute.
        input: Text to send to a waiting input() prompt.
        allow_input: If false, input() calls fail instead of waiting.

        Streams stdout/stderr as real-time notifications.
        """
        if code and input:
            return "ERROR: provide either code or input, not both"

        if input is not None:
            # ── provide input ──
            try:
                execution = service._current
                if not execution or execution.is_terminal:
                    return "ERROR: No active execution"
                if execution.status != "input_required" or not execution.input_request:
                    return "ERROR: No input request pending"

                rid = execution.input_request.inputRequestId
                service.provide_input(execution.execution_id, rid, input)
                await _stream_until_done(service, execution, ctx)

                if execution.status == "input_required":
                    return _fmt_input(service.get_execution(execution.execution_id))
                result = service.get_execution_result(execution.execution_id)
                _enrich(service, result)
                return _fmt_run(result)
            except Exception as e:
                return f"ERROR: {e}"

        if not code:
            return "ERROR: provide code to execute"

        # ── execute ──
        try:
            accepted = service.start_execution(code=code, allow_input=allow_input)
        except RuntimeError as e:
            return f"ERROR: {e}"

        exec_id = accepted["execution"]["executionId"]
        execution = service._executions[exec_id]

        await _stream_until_done(service, execution, ctx)

        if execution.status == "input_required":
            return _fmt_input(service.get_execution(exec_id))

        result = service.get_execution_result(exec_id)
        _enrich(service, result)
        return _fmt_run(result)

    # ── py_look ──────────────────────────────────────────────

    @mcp.tool
    def look(
        at: str | None = None,
        code: str | None = None,
        cursor: int | None = None,
    ) -> str:
        """See runtime state, inspect a symbol, or get completions.

        No args: overview with variable list.
        at="df": inspect the symbol df.
        at="df.columns": drill into nested attribute.
        code="df.hea" cursor=6: get completions at cursor position.
        """
        # ── completions ──
        if code is not None and cursor is not None:
            try:
                r = service.complete(code=code, cursor=cursor)
            except RuntimeError:
                r = {"matches": [], "cursorStart": cursor, "cursorEnd": cursor, "source": "runtime"}
            return _fmt_completions(r)

        # ── inspect ──
        if at is not None:
            return _fmt_inspect(_do_inspect(service, at), at)

        # ── overview ──
        health = service.get_health()
        try:
            variables = service.list_variables()
        except RuntimeError:
            variables = {"variables": [], "count": 0}
        return _fmt_overview(health, variables)

    # ── py_ctl ───────────────────────────────────────────────

    @mcp.tool
    def ctl(op: str, scope: str = "namespace") -> str:
        """Control the runtime.

        op="reset": clear namespace (scope: "namespace", "history", "assets", or "all").
        op="cancel": cancel the active execution.
        op="restart": restart interpreter (heavier than reset, fixes corrupted state).
        op="status": show detailed runtime health.
        """
        if op == "reset":
            scopes = ["namespace", "history", "assets"] if scope == "all" else [scope]
            try:
                r = service.reset(scope=scopes)
                return f"RESET | {', '.join(r.get('cleared', []))} cleared | 0 vars"
            except RuntimeError as e:
                return f"ERROR: {e}"
        elif op == "cancel":
            eid = service.get_health().get("currentExecutionId")
            if not eid:
                return "No active execution."
            try:
                service.cancel_execution(eid)
                return f"CANCELLED {eid}"
            except RuntimeError as e:
                return f"ERROR: {e}"
        elif op == "restart":
            try:
                service.shutdown()
                service.reset(scope=["namespace", "history", "assets"])
                return "RESTARTED | interpreter reinitialized | 0 vars"
            except Exception as e:
                return f"ERROR: {e}"
        elif op == "status":
            h = service.get_health()
            parts = [f"python {h.get('state', '?')}",
                     f"exec #{h.get('executionCount', 0)}",
                     f"rev {h.get('stateRevision', 0)}",
                     f"up {h.get('uptimeSeconds', 0)}s"]
            eid = h.get("currentExecutionId")
            if eid:
                parts.append(f"running: {eid}")
            return " | ".join(parts)
        return f"ERROR: unknown op '{op}'. Use reset, cancel, restart, or status."

    return mcp


def _do_inspect(service, symbol):
    """Inspect a symbol. Tries multiple strategies for best result.

    For variables: get_variable (with children) → list_variables → inspect → hover
    For dotted paths: get_variable with path → inspect → hover → eval via py
    """
    result = {"found": False}

    # Parse name and path: "data" → ("data", []),  "data.scores" → ("data", ["scores"])
    parts = symbol.split(".")
    root_name = parts[0]
    path = parts[1:] if len(parts) > 1 else None

    # 1. Try get_variable — gives children for structured drill-down
    if re.match(r"^[a-zA-Z_]\w*$", root_name):
        try:
            d = service.get_variable(name=root_name, path=path)
            if d.get("name") and d.get("type") != "error":
                d["found"] = True
                d["kind"] = "variable"
                result = d
        except (RuntimeError, KeyError):
            pass

    # 2. Try inspect/hover — richer for functions/modules (signature, docstring)
    for fn in [
        lambda: service.inspect(code=symbol, cursor=len(symbol), detail=1),
        lambda: service.hover(code=symbol, cursor=len(symbol)),
    ]:
        try:
            r = fn()
            if r.get("found"):
                if result.get("found"):
                    # Merge: keep variable detail (children, value), add inspect info
                    for key in ("signature", "docstring", "sourceCode", "file", "line"):
                        if r.get(key) and not result.get(key):
                            result[key] = r[key]
                    if r.get("kind") and r["kind"] != "variable":
                        result["kind"] = r["kind"]
                else:
                    result = r
                break
        except RuntimeError:
            pass

    # 3. Fallback to list_variables for simple names
    if (not result.get("found") or
            result.get("value") in ("<not found>", None) or
            result.get("type") in ("unknown", "error")):
        if re.match(r"^[a-zA-Z_]\w*$", symbol):
            try:
                vl = service.list_variables()
                for v in vl.get("variables", []):
                    if v.get("name") == symbol:
                        result["found"] = True
                        result["kind"] = "variable"
                        result["name"] = v["name"]
                        if result.get("value") in ("<not found>", None):
                            result["value"] = v.get("value")
                        if result.get("type") in ("unknown", "error", None):
                            result["type"] = v.get("type")
                        for k in ("size", "shape", "dtype", "length", "keys", "expandable"):
                            if v.get(k) is not None:
                                result[k] = v[k]
                        break
            except RuntimeError:
                pass

    # 4. Last resort for dotted paths: eval via execute
    if not result.get("found") and "." in symbol:
        try:
            er = service.execute_sync(
                code=f"_mrp_val = {symbol}\nprint(type(_mrp_val).__name__, ':', repr(_mrp_val)[:200])\ndel _mrp_val",
                store_history=False,
                allow_input=False,
            )
            if er.get("success"):
                stdout = er.get("stdout", "").strip()
                if ":" in stdout:
                    vtype, _, vval = stdout.partition(":")
                    result = {
                        "found": True,
                        "kind": "variable",
                        "name": symbol,
                        "type": vtype.strip(),
                        "value": vval.strip().strip("'\""),
                    }
        except Exception:
            pass

    # Clean up
    if result.get("value") == "<not found>":
        result.pop("value", None)

    return result


def _enrich(service, result):
    try:
        v = service.list_variables()
        result["_varCount"] = v.get("count", len(v.get("variables", [])))
    except Exception:
        pass
