"""
MRP Application — combined FastAPI host.

Serves:
- /mcp          — MCP Streamable HTTP (the primary API)
- /health       — plain HTTP health check
- /assets/{id}  — plain HTTP asset serving
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .service import RuntimeService
from .mcp_server import create_mcp_server


def create_app(
    cwd: str | None = None,
    assets_dir: str | None = None,
    venv: str | None = None,
) -> FastAPI:
    """Create the MRP application.

    Returns a FastAPI app with MCP mounted at /mcp and plain HTTP
    endpoints for health and assets.
    """
    service = RuntimeService(cwd=cwd, assets_dir=assets_dir, venv=venv)
    mcp = create_mcp_server(service=service)

    # Create the MCP ASGI app
    mcp_app = mcp.http_app(path="/mcp")

    # Create FastAPI host
    app = FastAPI(
        title="mrmd-python",
        version="0.4.0-draft",
        description="MRP runtime server for Python. Primary API at /mcp.",
        lifespan=mcp_app.lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount MCP
    app.mount("/mcp-server", mcp_app)

    # ── Plain HTTP endpoints ─────────────────────────────────────

    @app.get("/health")
    async def health():
        return service.get_health()

    @app.get("/assets/{asset_id}")
    async def get_asset(asset_id: str):
        asset_path = service.get_asset_path(asset_id)
        if asset_path is None:
            return JSONResponse(
                {"error": "asset_not_found", "message": f"Asset {asset_id} not found"},
                status_code=404,
            )

        suffix = asset_path.suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".html": "text/html",
            ".json": "application/json",
        }
        content_type = content_types.get(suffix, "application/octet-stream")
        return FileResponse(asset_path, media_type=content_type)

    return app
