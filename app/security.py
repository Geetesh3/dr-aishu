"""FastAPI dependencies that enforce API key authentication.

Two modes, controlled by ``DR_AISHU_REQUIRE_KEY``:

* ``0`` (default) - a request without a key is served under a per-IP "anonymous"
  demo key with tight limits. This is what makes the hosted demo and the browser
  UI work with no signup, while still throttling abuse.
* ``1`` - a valid key is mandatory; unauthenticated calls get 401. Use this when
  the instance is meant to be consumed only through the API.

Rate-limit state is reported on every response so clients can back off
correctly: ``X-RateLimit-Limit-Minute``, ``X-RateLimit-Remaining-Minute``,
``X-RateLimit-Limit-Day``, ``X-RateLimit-Remaining-Day``, plus ``Retry-After``
when a request is refused.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

from fastapi import Header, HTTPException, Request, Response, status

from . import auth

REQUIRE_KEY = os.environ.get("DR_AISHU_REQUIRE_KEY", "0").strip().lower() in {"1", "true", "yes", "on"}

# Anonymous keys are minted per client address and reused so limits accumulate.
_anon_lock = threading.Lock()
_anon_keys: dict[str, str] = {}


@dataclass
class AuthContext:
    key: auth.ApiKey
    limits: dict[str, object] = field(default_factory=dict)
    anonymous: bool = False


def _client_ip(request: Request) -> str:
    """Best-effort client address behind a proxy (Render, Fly, Colab tunnels)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    return request.client.host if request.client else "unknown"


def _extract_token(header_key: str | None, authorization: str | None) -> str | None:
    if header_key and header_key.strip():
        return header_key.strip()
    if authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            return value[7:].strip()
    return None


def _anonymous_key(ip: str) -> auth.ApiKey:
    with _anon_lock:
        key_id = _anon_keys.get(ip)
        if key_id:
            existing = auth.get_key(key_id)
            if existing and not existing.revoked:
                return existing
        _plaintext, key = auth.create_key(tier="anonymous", label=f"auto:{ip}", ip=ip)
        _anon_keys[ip] = key.id
        return key


def reset_anonymous_cache() -> None:
    """Test helper."""
    with _anon_lock:
        _anon_keys.clear()


def _apply_headers(response: Response, limits: dict) -> None:
    minute = limits.get("minute") or {}
    day = limits.get("day") or {}
    if minute:
        response.headers["X-RateLimit-Limit-Minute"] = str(minute.get("limit"))
        response.headers["X-RateLimit-Remaining-Minute"] = str(minute.get("remaining"))
    if day:
        response.headers["X-RateLimit-Limit-Day"] = str(day.get("limit"))
        response.headers["X-RateLimit-Remaining-Day"] = str(day.get("remaining"))
    response.headers["X-RateLimit-Tier"] = str(limits.get("tier", "free"))


def apply_headers_from_state(request: Request, response: Response) -> None:
    """Copy rate-limit headers recorded by the dependency onto the final response.

    Endpoints that return a ``JSONResponse`` directly ignore headers set on the
    injected ``Response``, so the values are stashed on ``request.state`` and
    replayed by the middleware in ``main``.
    """
    limits = getattr(request.state, "rate_limits", None)
    if limits:
        _apply_headers(response, limits)


async def rate_limit_middleware(request: Request, call_next):
    response = await call_next(request)
    apply_headers_from_state(request, response)
    return response


def require_scope(scope: str | None):
    """Build a dependency that authenticates a key and checks one scope.

    Pass ``None`` to authenticate and rate-limit a caller without demanding a
    particular scope. Self-inspection endpoints such as ``/api/keys/me`` need
    this: a caller must be able to read its own scopes even when its tier does
    not grant the ``keys`` scope, otherwise the web UI cannot tell which actions
    to offer.
    """

    async def _dependency(
        request: Request,
        response: Response,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        authorization: str | None = Header(default=None),
    ) -> AuthContext:
        token = _extract_token(x_api_key, authorization)
        anonymous = False

        if token:
            key = auth.find_by_plaintext(token)
            if key is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key. Create one at /developers.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            if key.revoked:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="This API key has been revoked.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        elif REQUIRE_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required. Send it in the X-API-Key header. "
                       "Create one at /developers.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            key = _anonymous_key(_client_ip(request))
            anonymous = True

        if scope is not None and scope not in key.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This key's tier ({key.tier}) does not include the '{scope}' scope.",
            )

        allowed, limits = auth.consume(key)
        if not allowed:
            _apply_headers(response, limits)
            request.state.rate_limits = limits
            window = limits.get("blocked_window", "minute")
            retry = 60 if window == "minute" else 3600
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit reached ({window} window). "
                       f"Tier '{key.tier}' allows "
                       f"{key.limits['per_minute']}/min and {key.limits['per_day']}/day.",
                headers={"Retry-After": str(retry)},
            )

        _apply_headers(response, limits)
        request.state.rate_limits = limits
        return AuthContext(key=key, limits=limits, anonymous=anonymous)

    return _dependency


# Ready-made dependencies for the endpoint declarations.
require_analyze = require_scope(auth.SCOPE_ANALYZE)
require_read = require_scope(auth.SCOPE_READ)
require_feedback = require_scope(auth.SCOPE_FEEDBACK)
require_export = require_scope(auth.SCOPE_EXPORT)
require_train = require_scope(auth.SCOPE_TRAIN)
require_keys = require_scope(auth.SCOPE_KEYS)

# The operator scope gates deployment internals rather than clinical functions:
# the dataset browser (build artefacts, archive hashes, exact class counts) and
# retraining. Both are the same tier's job, so they share one scope; a public
# visitor must not be able to enumerate how the instance was built.
require_operator = require_scope(auth.SCOPE_TRAIN)
# Any valid key (or the browser's anonymous key) may identify itself.
require_identity = require_scope(None)
