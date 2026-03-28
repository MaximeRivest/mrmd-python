"""
MRP MCP Server — the primary interface for mrmd-python.

Defines MCP tools, resources, and task support over a shared RuntimeService.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from .service import RuntimeService


def create_mcp_server(
    service: RuntimeService | None = None,
    name: str = "mrmd-python",
    **service_kwargs,
) -> FastMCP:
    """Create an MCP server backed by a RuntimeService."""

    if service is None:
        service = RuntimeService(**service_kwargs)

    @asynccontextmanager
    async def lifespan(server):
        yield
        service.shutdown()

    mcp = FastMCP(
        name=name,
        instructions=(
            "Live Python runtime. Use runtime_execute to run code. "
            "Use runtime_list_variables, runtime_get_variable, runtime_hover, "
            "and runtime_inspect for live namespace introspection. "
            "Use runtime_reset to clear state."
        ),
        lifespan=lifespan,
    )

    # ── Resources ────────────────────────────────────────────────

    @mcp.resource("mrp://health")
    def health_resource() -> str:
        """Runtime health and current state."""
        import json
        return json.dumps(service.get_health())

    @mcp.resource("mrp://capabilities")
    def capabilities_resource() -> str:
        """Runtime capabilities and feature flags."""
        import json
        return json.dumps(service.get_capabilities())

    # ── Runtime State Tools ──────────────────────────────────────

    @mcp.tool
    def runtime_get_health() -> dict:
        """Read lightweight runtime liveness and state."""
        return service.get_health()

    @mcp.tool
    def runtime_get_capabilities() -> dict:
        """Read runtime capabilities and feature flags."""
        return service.get_capabilities()

    @mcp.tool
    def runtime_reset(scope: list[str] | None = None) -> dict:
        """Clear selected runtime state. Scope: namespace, history, assets."""
        return service.reset(scope=scope)

    # ── Execution Tools ──────────────────────────────────────────

    @mcp.tool
    def runtime_execute(
        code: str,
        store_history: bool = True,
        silent: bool = False,
        allow_input: bool = True,
        client_execution_id: str | None = None,
    ) -> dict:
        """Execute code and return the final result.

        This is the primary execution tool. For short code, it returns
        the complete result. For long-running code, use with MCP task
        augmentation.
        """
        return service.execute_sync(
            code=code,
            store_history=store_history,
            silent=silent,
            allow_input=allow_input,
            client_execution_id=client_execution_id,
        )

    @mcp.tool
    def runtime_start_execution(
        code: str,
        store_history: bool = True,
        silent: bool = False,
        allow_input: bool = True,
        client_execution_id: str | None = None,
    ) -> dict:
        """Start a detached execution and return immediately with an execution handle.

        Use runtime_get_execution, runtime_get_execution_events, and
        runtime_get_execution_result to observe and retrieve the result.
        """
        return service.start_execution(
            code=code,
            store_history=store_history,
            silent=silent,
            allow_input=allow_input,
            client_execution_id=client_execution_id,
        )

    @mcp.tool
    def runtime_get_execution(execution_id: str) -> dict:
        """Read the current state of an execution."""
        return service.get_execution(execution_id)

    @mcp.tool
    def runtime_get_execution_events(
        execution_id: str,
        after_seq: int = 0,
        limit: int = 100,
    ) -> dict:
        """Read incremental execution events (stdout, stderr, display, assets, warnings)."""
        return service.get_execution_events(
            execution_id=execution_id,
            after_seq=after_seq,
            limit=limit,
        )

    @mcp.tool
    def runtime_get_execution_result(execution_id: str) -> dict:
        """Read the final result of a terminal execution."""
        return service.get_execution_result(execution_id)

    @mcp.tool
    def runtime_provide_input(
        execution_id: str,
        input_request_id: str,
        text: str,
    ) -> dict:
        """Provide text input to an execution waiting for input."""
        return service.provide_input(execution_id, input_request_id, text)

    @mcp.tool
    def runtime_cancel_input(
        execution_id: str,
        input_request_id: str,
    ) -> dict:
        """Cancel a pending input request."""
        return service.cancel_input(execution_id, input_request_id)

    @mcp.tool
    def runtime_cancel_execution(
        execution_id: str,
        reason: str | None = None,
    ) -> dict:
        """Cancel an active execution."""
        return service.cancel_execution(execution_id, reason)

    # ── Language Intelligence Tools ──────────────────────────────

    @mcp.tool
    def runtime_complete(
        code: str,
        cursor: int,
        trigger_kind: str = "invoked",
        trigger_character: str | None = None,
    ) -> dict:
        """Get completions at cursor position using live runtime state."""
        return service.complete(code=code, cursor=cursor)

    @mcp.tool
    def runtime_inspect(
        code: str,
        cursor: int,
        detail: int = 1,
    ) -> dict:
        """Get detailed information about a symbol at cursor position."""
        return service.inspect(code=code, cursor=cursor, detail=detail)

    @mcp.tool
    def runtime_hover(
        code: str,
        cursor: int,
    ) -> dict:
        """Get lightweight hover/tooltip information about a symbol."""
        return service.hover(code=code, cursor=cursor)

    # ── Variable Tools ───────────────────────────────────────────

    @mcp.tool
    def runtime_list_variables(
        max_value_length: int = 200,
    ) -> dict:
        """List variables in the runtime namespace."""
        return service.list_variables(maxValueLength=max_value_length)

    @mcp.tool
    def runtime_get_variable(
        name: str,
        path: list[str] | None = None,
        max_children: int = 100,
        max_value_length: int = 1000,
    ) -> dict:
        """Get detailed information about a variable, optionally drilling into children."""
        return service.get_variable(
            name=name,
            path=path,
            maxChildren=max_children,
            maxValueLength=max_value_length,
        )

    # ── Code Analysis Tools ──────────────────────────────────────

    @mcp.tool
    def runtime_is_complete(code: str) -> dict:
        """Check whether code is a complete statement."""
        return service.is_complete(code=code)

    @mcp.tool
    def runtime_format(code: str) -> dict:
        """Format code using the runtime's formatter (e.g., black for Python)."""
        return service.format_code(code=code)

    # ── History Tool ─────────────────────────────────────────────

    @mcp.tool
    def runtime_query_history(
        n: int = 20,
        pattern: str | None = None,
        before: int | None = None,
    ) -> dict:
        """Read execution input history."""
        return service.query_history(n=n, pattern=pattern, before=before)

    return mcp
