"""FastAPI application entry point.

Serves the REST API under ``/api``, WebSocket events under ``/ws``, and the
compiled React frontend from ``src/orinode/ui/static/`` under ``/``.

When ``ORINODE_UI_USER`` and ``ORINODE_UI_PASS`` are set, all requests
(including static files and WebSockets) require HTTP Basic authentication.

Bind address is controlled via environment variables:
  ORINODE_UI_HOST  — default 127.0.0.1 (localhost only, no external access)
  ORINODE_UI_PORT  — default 7860

Usage::

    python -m orinode.ui.server          # production (after make ui-build)
    uvicorn orinode.ui.server:app --reload  # dev (hot-reload Python only)
"""

from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from orinode.ui.api import router
from orinode.ui.api_v1 import router as router_v1
from orinode.ui.websocket import ws_run_events

_STATIC_DIR = Path(__file__).parent / "static"

HOST = os.environ.get("ORINODE_UI_HOST", "127.0.0.1")
PORT = int(os.environ.get("ORINODE_UI_PORT", "7860"))

UI_USERNAME = os.environ.get("ORINODE_UI_USER", "")
UI_PASSWORD = os.environ.get("ORINODE_UI_PASS", "")

_AUTH_ENABLED = bool(UI_USERNAME and UI_PASSWORD)

_UNAUTH = JSONResponse(
    status_code=401,
    content={"detail": "Authentication required"},
    headers={"WWW-Authenticate": 'Basic realm="Orinode-LM"'},
)


def _check_basic(auth_header: str) -> bool:
    if not auth_header.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8", errors="strict")
        username, _, password = decoded.partition(":")
        return secrets.compare_digest(username, UI_USERNAME) and secrets.compare_digest(
            password, UI_PASSWORD
        )
    except Exception:
        return False


app = FastAPI(
    title="Orinode-LM Dashboard",
    description="Training dashboard for the Nigerian Speech-LLM",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        f"http://localhost:{PORT}",
        f"http://{HOST}:{PORT}",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    if not _AUTH_ENABLED:
        return await call_next(request)
    if not _check_basic(request.headers.get("authorization", "")):
        return _UNAUTH
    return await call_next(request)


app.include_router(router, prefix="/api")
app.include_router(router_v1, prefix="/api/v1")


@app.websocket("/ws/runs/{run_id}")
async def ws_runs_auth(websocket, run_id: str) -> None:  # type: ignore[no-untyped-def]
    if _AUTH_ENABLED and not _check_basic(websocket.headers.get("authorization", "")):
        await websocket.close(code=1008, reason="Auth required")
        return
    await ws_run_events(websocket, run_id)


# Serve compiled React frontend if it has been built
if _STATIC_DIR.exists() and any(_STATIC_DIR.iterdir()):
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:  # noqa: ARG001
        return FileResponse(_STATIC_DIR / "index.html")


def main() -> None:
    import uvicorn

    from orinode.paths import ensure_workspace

    ensure_workspace()
    uvicorn.run(
        "orinode.ui.server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
