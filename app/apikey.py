"""
API-key gate for machine-to-machine access (the MCP server + the media/export routes an
external agent needs to fetch images and clips).

This sits OUTSIDE the Google-SSO `AuthGate` (installed last in server.py, so it runs first):

  - `/mcp...` : an API key is REQUIRED. No/invalid key -> 401.
  - media + export routes (`/frames`, `/clips`, `/exports`, `/api/evidence/export...`) : a valid
    API key GRANTS access, so a key-bearing agent can fetch the images/clips a tool points it at
    without a browser session. Without a key these fall through to the SSO gate, so logged-in
    humans are unaffected.

When a valid key is presented, the gate sets `scope['state']['mcp_authed'] = True`; the SSO
`AuthGate` checks that flag and passes the request straight through. Keys are managed at /keys.
"""
from __future__ import annotations
import logging
from urllib.parse import parse_qs

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import config, db

log = logging.getLogger(__name__)

_MCP_PREFIX = "/mcp"
# Routes a valid key may access without a browser session.
_MEDIA_PREFIXES = ("/frames/", "/clips/", "/exports/", "/api/evidence/export")


def _is_mcp(path: str) -> bool:
    return path == _MCP_PREFIX or path.startswith(_MCP_PREFIX + "/")


def _is_media(path: str) -> bool:
    return any(path.startswith(p) for p in _MEDIA_PREFIXES)


def _extract_key(scope) -> str | None:
    headers = dict(scope.get("headers") or [])
    auth = headers.get(b"authorization")
    if auth:
        v = auth.decode("latin-1").strip()
        if v.lower().startswith("bearer "):
            return v[7:].strip()
    xk = headers.get(b"x-api-key")
    if xk:
        return xk.decode("latin-1").strip()
    # Query-param fallback (?key=) for embedding media URLs where headers are awkward.
    qs = scope.get("query_string", b"")
    if qs:
        vals = parse_qs(qs.decode("latin-1")).get("key")
        if vals:
            return vals[0]
    return None


class McpKeyGate:
    """Pure-ASGI key gate. Pure ASGI (not BaseHTTPMiddleware) so it never buffers responses or
    breaks the Range requests that video-clip seeking relies on."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        is_mcp, is_media = _is_mcp(path), _is_media(path)
        if not (is_mcp or is_media):
            await self.app(scope, receive, send)
            return

        rec = await db.verify_api_key(_extract_key(scope))
        if rec:
            state = scope.setdefault("state", {})
            state["mcp_authed"] = True
            state["api_key"] = rec
            if is_mcp:
                await db.touch_api_key(rec["id"])  # don't write on every media fetch
            await self.app(scope, receive, send)
            return

        if is_mcp:
            resp = JSONResponse({"error": "valid API key required"}, status_code=401)
            await resp(scope, receive, send)
            return

        # Media/export with no key → let the SSO gate decide (a logged-in human still works).
        await self.app(scope, receive, send)


def install(app: FastAPI) -> None:
    if not config.MCP_ENABLED:
        return
    # Added after the SSO middleware in server.py, so this ends up OUTERMOST and runs first.
    app.add_middleware(McpKeyGate)
