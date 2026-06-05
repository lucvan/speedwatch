"""
Google SSO authentication + role-based authorisation for speedwatch.

The UI is exposed publicly (https://speedwatch.e49ta.com via the Synology reverse proxy),
so every request is gated:

  1. Identity — Google OpenID Connect. We use the ID token once at login to learn the
     user's email, then issue our own signed-cookie session (we never store Google tokens).
  2. Allowlist — the email must exist in the `app_user` table (strict allowlist). Google
     authenticating an account is NOT enough; an admin must have added it.
  3. Role — readonly < edit < admin, enforced per request by method + path (see
     `required_role`). New routes are protected by default (fail-closed middleware).

Wiring: `auth.install(app, templates)` from server.py adds the middleware (session +
auth), registers the /login, /auth/callback, /logout and /users routes, and exposes a
`current_user(request)` Jinja global.
"""
from __future__ import annotations
import logging

from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from . import config, db

log = logging.getLogger(__name__)

# Role ordering — a user satisfies a requirement if their rank is >= the required rank.
ROLE_RANK = {"readonly": 0, "edit": 1, "admin": 2}
READONLY, EDIT, ADMIN = 0, 1, 2

# Paths reachable without a session (login flow + static assets + health).
_PUBLIC_PREFIXES = ("/login", "/auth/", "/logout", "/static/", "/healthz", "/favicon.ico")

oauth = OAuth()


def role_rank(role: str | None) -> int:
    return ROLE_RANK.get(role or "", -1)


def _is_public(path: str) -> bool:
    return any(path == p or path.startswith(p) for p in _PUBLIC_PREFIXES)


def required_role(method: str, path: str) -> int:
    """Minimum role rank needed for (method, path).

    - User management is admin-only.
    - Anything destructive (DELETE) or that rewrites calibration / masks is admin-only.
    - Any other mutating request (POST/PUT/PATCH) needs edit.
    - Everything else (GET/HEAD page + read API) needs only readonly.
    """
    if path == "/users" or path.startswith("/users/") or path.startswith("/api/users"):
        return ADMIN
    if method == "DELETE":
        return ADMIN
    if (path.startswith("/api/calibration")
            or path.startswith("/api/lens/save")
            or path.startswith("/api/anpr-mask")):
        return ADMIN
    if method in ("POST", "PUT", "PATCH"):
        return EDIT
    return READONLY


def current_user(request: Request) -> dict | None:
    """The authenticated+authorised user for this request, or None. Set by the middleware."""
    return getattr(request.state, "user", None)


def _accept_header(scope) -> str:
    for k, v in scope.get("headers", []):
        if k == b"accept":
            return v.decode("latin-1")
    return ""


def _wants_json(scope, path: str) -> bool:
    """API/fetch callers should get a JSON 401/403; browsers navigating get a redirect."""
    if path.startswith("/api/"):
        return True
    if scope["method"] not in ("GET", "HEAD"):
        return True
    accept = _accept_header(scope)
    return "application/json" in accept and "text/html" not in accept


class AuthGate:
    """Pure-ASGI auth/authorisation gate. Pure ASGI (not BaseHTTPMiddleware) so it never
    buffers responses or breaks the Range requests that video-clip seeking relies on.

    Runs INSIDE SessionMiddleware, so `scope['session']` is already populated. Sets
    `scope['state']['user']` so downstream request.state.user / templates can read it."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope["path"]
        state = scope.setdefault("state", {})
        state["user"] = None
        if _is_public(path):
            await self.app(scope, receive, send)
            return

        session = scope.get("session", {})
        email = session.get("user_email")
        user = await db.get_user(email) if email else None

        if user is None:
            if email:
                session.pop("user_email", None)  # stale session for a removed user
            if _wants_json(scope, path):
                resp = JSONResponse({"error": "authentication required"}, status_code=401)
            else:
                resp = RedirectResponse(url="/login", status_code=303)
            await resp(scope, receive, send)
            return

        state["user"] = user
        if role_rank(user["role"]) < required_role(scope["method"], path):
            if _wants_json(scope, path):
                resp = JSONResponse(
                    {"error": "insufficient permissions", "role": user["role"]},
                    status_code=403,
                )
            else:
                resp = HTMLResponse(_forbidden_html(user), status_code=403)
            await resp(scope, receive, send)
            return

        await self.app(scope, receive, send)


def install(app: FastAPI, templates) -> None:
    # Make the current user available to every template (nav, footer, Users link).
    templates.env.globals["current_user"] = current_user

    if not config.AUTH_ENABLED:
        log.warning(
            "\n" + "!" * 74 + "\n"
            "!! AUTHENTICATION DISABLED — GOOGLE_CLIENT_ID is not set in .env.\n"
            "!! The UI is served WITHOUT login: anyone who can reach it has full access.\n"
            "!! This is only safe on localhost for development. Configure GOOGLE_CLIENT_ID,\n"
            "!! GOOGLE_CLIENT_SECRET, SESSION_SECRET and BOOTSTRAP_ADMIN before exposing it.\n"
            + "!" * 74
        )
        return

    if not config.SESSION_SECRET:
        raise RuntimeError("SESSION_SECRET must be set when authentication is enabled")

    oauth.register(
        name="google",
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=config.GOOGLE_CLIENT_ID,
        client_secret=config.GOOGLE_CLIENT_SECRET,
        client_kwargs={"scope": "openid email profile"},
    )

    # ── Middleware ──────────────────────────────────────────────────────────────
    # Added inner-first: the AuthGate is registered first, SessionMiddleware second so it
    # ends up outermost and populates scope['session'] before the gate reads it.
    app.add_middleware(AuthGate)
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.SESSION_SECRET,
        https_only=True,
        same_site="lax",
        max_age=config.SESSION_MAX_AGE,
    )

    # ── Auth routes ─────────────────────────────────────────────────────────────
    @app.get("/login", response_class=HTMLResponse)
    async def login_page(request: Request):
        # Already signed in → straight to the app.
        if request.session.get("user_email"):
            return RedirectResponse(url="/", status_code=303)
        return templates.TemplateResponse("login.html", {"request": request})

    @app.get("/auth/login")
    async def auth_login(request: Request):
        return await oauth.google.authorize_redirect(request, config.OAUTH_REDIRECT_URI)

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        try:
            token = await oauth.google.authorize_access_token(request)
        except Exception as e:
            log.warning("OAuth callback failed: %s", e)
            return templates.TemplateResponse(
                "login.html", {"request": request, "error": "Sign-in failed. Please try again."},
                status_code=400,
            )
        info = token.get("userinfo") or {}
        email = (info.get("email") or "").lower()
        if not email or not info.get("email_verified", True):
            return templates.TemplateResponse(
                "login.html",
                {"request": request, "error": "Google did not return a verified email."},
                status_code=400,
            )
        user = await db.get_user(email)
        if user is None:
            # Strict allowlist: authenticated, but not authorised.
            return templates.TemplateResponse(
                "denied.html", {"request": request, "email": email}, status_code=403,
            )
        await db.touch_last_login(email, name=info.get("name"))
        request.session["user_email"] = email
        log.info("login: %s (%s)", email, user["role"])
        return RedirectResponse(url="/", status_code=303)

    @app.get("/logout")
    async def logout(request: Request):
        request.session.pop("user_email", None)
        return RedirectResponse(url="/login", status_code=303)

    # ── User management (admin only — enforced by the middleware) ────────────────
    @app.get("/users", response_class=HTMLResponse)
    async def users_page(request: Request):
        users = await db.list_users()
        return templates.TemplateResponse("users.html", {
            "request": request,
            "users": users,
            "roles": db.VALID_ROLES,
            "bootstrap_admin": config.BOOTSTRAP_ADMIN,
            "pipeline_status": "running",
        })

    @app.post("/api/users")
    async def api_upsert_user(
        request: Request,
        email: str = Form(...),
        role: str = Form(...),
    ):
        email = email.strip().lower()
        role = role.strip().lower()
        if "@" not in email or "." not in email.split("@")[-1]:
            raise HTTPException(status_code=422, detail="Enter a valid email address")
        if role not in db.VALID_ROLES:
            raise HTTPException(status_code=422, detail="Invalid role")
        # Don't let the last admin (or the bootstrap admin) be demoted into lockout.
        existing = await db.get_user(email)
        if existing and existing["role"] == "admin" and role != "admin":
            if email == config.BOOTSTRAP_ADMIN:
                raise HTTPException(status_code=400, detail="The bootstrap admin cannot be demoted")
            if await db.count_admins() <= 1:
                raise HTTPException(status_code=400, detail="Cannot demote the last admin")
        actor = current_user(request)
        await db.upsert_user(email, role, added_by=actor["email"] if actor else None)
        return JSONResponse({"ok": True, "email": email, "role": role})

    @app.post("/api/users/delete")
    async def api_delete_user(request: Request, email: str = Form(...)):
        email = email.strip().lower()
        if email == config.BOOTSTRAP_ADMIN:
            raise HTTPException(status_code=400, detail="The bootstrap admin cannot be removed")
        target = await db.get_user(email)
        if target and target["role"] == "admin" and await db.count_admins() <= 1:
            raise HTTPException(status_code=400, detail="Cannot remove the last admin")
        actor = current_user(request)
        if actor and actor["email"] == email:
            raise HTTPException(status_code=400, detail="You cannot remove your own account")
        await db.delete_user(email)
        return JSONResponse({"ok": True})


def _forbidden_html(user: dict) -> str:
    return (
        "<!doctype html><meta charset='utf-8'><title>Forbidden</title>"
        "<body style='font-family:system-ui;background:#111;color:#e0e0e0;"
        "display:flex;min-height:100vh;align-items:center;justify-content:center;text-align:center'>"
        "<div><h1 style='color:#ff6b6b'>403 — Not allowed</h1>"
        f"<p>Your account (<code>{user['email']}</code>, role <b>{user['role']}</b>) "
        "doesn't have permission for that action.</p>"
        "<p><a href='/' style='color:#4fc3f7'>Back to dashboard</a></p></div></body>"
    )
