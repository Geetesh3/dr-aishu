"""API key issuance, verification and rate limiting for Dr. Aishu.

Design notes
------------
* Keys are high-entropy random tokens, so a plain SHA-256 digest is the right
  storage form. Only the digest is ever written to disk; the plaintext is shown
  to the developer exactly once, at creation time.
* Lookups are by digest, which is a constant-time indexed comparison and avoids
  leaking key material through timing.
* Rate limits are enforced per key across two windows (minute and day). A
  request is rejected only if some window is exhausted, and counters are
  incremented only when the request is admitted, so a throttled call does not
  burn the caller's daily quota.
* Tiers carry a scope set. Endpoints declare the scope they need, so a demo key
  can analyse symptoms but can never trigger a retrain.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from . import store

KEY_PREFIX = "dra_"
_HASH_BYTES = 32

# Scope names used by the endpoint dependencies.
SCOPE_ANALYZE = "analyze"
SCOPE_READ = "read"
SCOPE_FEEDBACK = "feedback"
SCOPE_EXPORT = "export"
SCOPE_TRAIN = "train"
SCOPE_KEYS = "keys"

TIERS: dict[str, dict[str, Any]] = {
    "anonymous": {
        "label": "Demo (browser)",
        "per_minute": 6,
        "per_day": 30,
        "scopes": {SCOPE_ANALYZE, SCOPE_READ, SCOPE_FEEDBACK},
        "description": "Automatic key for the web interface. Enough to try Dr. Aishu.",
    },
    "free": {
        "label": "Free",
        "per_minute": 20,
        "per_day": 500,
        "scopes": {SCOPE_ANALYZE, SCOPE_READ, SCOPE_FEEDBACK, SCOPE_KEYS},
        "description": "For personal projects and prototypes. No cost, no card.",
    },
    "pro": {
        "label": "Pro",
        "per_minute": 120,
        "per_day": 20000,
        "scopes": {SCOPE_ANALYZE, SCOPE_READ, SCOPE_FEEDBACK, SCOPE_KEYS, SCOPE_EXPORT, SCOPE_TRAIN},
        "description": "Operator tier: training and data export. Issued only by the operator.",
    },
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id            TEXT PRIMARY KEY,
    key_hash      TEXT NOT NULL UNIQUE,
    display       TEXT NOT NULL,
    tier          TEXT NOT NULL,
    email         TEXT,
    label         TEXT,
    created_at    TEXT NOT NULL,
    last_used_at  TEXT,
    revoked       INTEGER DEFAULT 0,
    request_count INTEGER DEFAULT 0,
    ip            TEXT
);
CREATE INDEX IF NOT EXISTS idx_keys_hash   ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_keys_email  ON api_keys (email);

CREATE TABLE IF NOT EXISTS api_usage (
    key_id  TEXT NOT NULL,
    window  TEXT NOT NULL,
    bucket  TEXT NOT NULL,
    count   INTEGER DEFAULT 0,
    PRIMARY KEY (key_id, window, bucket)
);
CREATE INDEX IF NOT EXISTS idx_usage_bucket ON api_usage (bucket);
"""

_lock = threading.Lock()
_ready: set[str] = set()


def init_db() -> None:
    """Create the auth tables. Idempotent and safe to call repeatedly."""
    with _lock, store._connect() as conn:
        conn.executescript(SCHEMA)
        # Usage buckets are ephemeral; drop anything older than two days.
        cutoff = (_now() - timedelta(days=2)).isoformat(timespec="seconds")
        conn.execute("DELETE FROM api_usage WHERE bucket < ?", (cutoff,))
        _ready.add(str(store.DB_PATH))


def _ensure() -> None:
    """Lazily create the schema for the current database.

    The store's path is patchable, so tests and multi-tenant runs can point at a
    fresh file at any time; tracking the resolved path keeps the tables present
    without relying on import-time ordering.
    """
    if str(store.DB_PATH) not in _ready:
        init_db()


@dataclass
class ApiKey:
    id: str
    display: str
    tier: str
    email: str | None = None
    label: str | None = None
    created_at: str = ""
    last_used_at: str | None = None
    revoked: bool = False
    request_count: int = 0
    scopes: set[str] = field(default_factory=set)

    @property
    def limits(self) -> dict[str, Any]:
        return TIERS.get(self.tier, TIERS["free"])

    def to_public(self) -> dict[str, Any]:
        """Safe to return to the holder: never includes the digest."""
        return {
            "id": self.id,
            "key": self.display,
            "tier": self.tier,
            "tier_label": self.limits["label"],
            "email": self.email,
            "label": self.label,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "revoked": self.revoked,
            "request_count": self.request_count,
            "scopes": sorted(self.scopes),
            "limits": {
                "per_minute": self.limits["per_minute"],
                "per_day": self.limits["per_day"],
            },
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def _display(plaintext: str) -> str:
    """A recognisable but non-reversible label, e.g. `dra_a1b2c3d4…`."""
    return f"{plaintext[:12]}…{plaintext[-4:]}"


def generate_plaintext() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(_HASH_BYTES)


def _row_to_key(row) -> ApiKey:
    tier = row["tier"] if row["tier"] in TIERS else "free"
    return ApiKey(
        id=row["id"],
        display=row["display"],
        tier=tier,
        email=row["email"],
        label=row["label"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        revoked=bool(row["revoked"]),
        request_count=row["request_count"] or 0,
        scopes=set(TIERS[tier]["scopes"]),
    )


def create_key(
    tier: str = "free",
    email: str | None = None,
    label: str | None = None,
    ip: str | None = None,
) -> tuple[str, ApiKey]:
    """Issue a new key. Returns (plaintext, record). The plaintext is not stored."""
    _ensure()
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")
    plaintext = generate_plaintext()
    key_id = secrets.token_hex(8)
    created = _iso(_now())
    digest = hash_key(plaintext)
    with _lock, store._connect() as conn:
        conn.execute(
            """INSERT INTO api_keys (id, key_hash, display, tier, email, label,
                   created_at, revoked, request_count, ip)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?)""",
            (key_id, digest, _display(plaintext), tier, email, label, created, ip),
        )
        row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
    return plaintext, _row_to_key(row)


def get_key(key_id: str) -> ApiKey | None:
    _ensure()
    with _lock, store._connect() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
    return _row_to_key(row) if row else None


def find_by_plaintext(plaintext: str) -> ApiKey | None:
    _ensure()
    if not plaintext:
        return None
    digest = hash_key(plaintext.strip())
    with _lock, store._connect() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (digest,)).fetchone()
    if row is None:
        return None
    # Defensive: the index lookup already matched, but confirm the digest in
    # constant time so a corrupted row can never authenticate.
    if not hmac.compare_digest(row["key_hash"], digest):
        return None
    return _row_to_key(row)


def _bucket(now: datetime, window: str) -> str:
    if window == "minute":
        return now.strftime("%Y-%m-%dT%H:%M")
    return now.strftime("%Y-%m-%d")


def consume(key: ApiKey, now: datetime | None = None) -> tuple[bool, dict[str, Any]]:
    """Charge one request against the key's windows.

    Returns (allowed, detail). Counters advance only when the request is allowed.
    """
    _ensure()
    now = now or _now()
    limits = TIERS[key.tier]
    windows = (("minute", limits["per_minute"]), ("day", limits["per_day"]))

    with _lock, store._connect() as conn:
        state: dict[str, Any] = {}
        blocked: str | None = None
        for window, limit in windows:
            bucket = _bucket(now, window)
            row = conn.execute(
                "SELECT count FROM api_usage WHERE key_id=? AND window=? AND bucket=?",
                (key.id, window, bucket),
            ).fetchone()
            used = row["count"] if row else 0
            state[window] = {"used": used, "limit": limit, "remaining": max(0, limit - used)}
            if used >= limit:
                blocked = window

        if blocked:
            return False, {
                "allowed": False,
                "blocked_window": blocked,
                **state,
                "tier": key.tier,
            }

        for window, _limit in windows:
            bucket = _bucket(now, window)
            conn.execute(
                """INSERT INTO api_usage (key_id, window, bucket, count)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(key_id, window, bucket) DO UPDATE SET count = count + 1""",
                (key.id, window, bucket),
            )
            state[window]["used"] += 1
            state[window]["remaining"] = max(0, state[window]["remaining"] - 1)

        conn.execute(
            "UPDATE api_keys SET last_used_at = ?, request_count = request_count + 1 WHERE id = ?",
            (_iso(now), key.id),
        )

    return True, {"allowed": True, **state, "tier": key.tier}


def usage(key_id: str) -> dict[str, Any]:
    _ensure()
    now = _now()
    with _lock, store._connect() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        if row is None:
            return {}
        out: dict[str, Any] = {}
        for window in ("minute", "day"):
            bucket = _bucket(now, window)
            used = conn.execute(
                "SELECT count FROM api_usage WHERE key_id=? AND window=? AND bucket=?",
                (key_id, window, bucket),
            ).fetchone()
            out[window] = {
                "used": used["count"] if used else 0,
                "limit": TIERS[row["tier"]]["per_minute" if window == "minute" else "per_day"],
            }
    for window, limit in (("minute", TIERS[row["tier"]]["per_minute"]), ("day", TIERS[row["tier"]]["per_day"])):
        out[window]["remaining"] = max(0, limit - out[window]["used"])
    return out


def revoke(key_id: str, email: str | None = None) -> bool:
    """Revoke a key. When `email` is given the key must belong to that owner."""
    _ensure()
    with _lock, store._connect() as conn:
        if email is None:
            cur = conn.execute("UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,))
        else:
            cur = conn.execute(
                "UPDATE api_keys SET revoked = 1 WHERE id = ? AND email = ?", (key_id, email)
            )
        return cur.rowcount > 0


def list_keys(email: str) -> list[dict[str, Any]]:
    _ensure()
    with _lock, store._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM api_keys WHERE email = ? ORDER BY created_at DESC", (email,)
        ).fetchall()
    return [_row_to_key(r).to_public() for r in rows]


def delete_key(key_id: str) -> bool:
    _ensure()
    with _lock, store._connect() as conn:
        conn.execute("DELETE FROM api_usage WHERE key_id = ?", (key_id,))
        cur = conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
        return cur.rowcount > 0


def stats() -> dict[str, Any]:
    _ensure()
    with _lock, store._connect() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM api_keys").fetchone()["c"]
        active = conn.execute("SELECT COUNT(*) c FROM api_keys WHERE revoked = 0").fetchone()["c"]
        by_tier = conn.execute(
            "SELECT tier, COUNT(*) c FROM api_keys WHERE revoked = 0 GROUP BY tier"
        ).fetchall()
        requests = conn.execute("SELECT COALESCE(SUM(request_count),0) c FROM api_keys").fetchone()["c"]
    return {
        "keys_total": total,
        "keys_active": active,
        "by_tier": {r["tier"]: r["c"] for r in by_tier},
        "requests_total": requests,
    }


# ---------------------------------------------------------------------------
# Per-IP throttle for anonymous key issuance
# ---------------------------------------------------------------------------
_IP_WINDOW_SECONDS = 3600
_IP_MAX_KEYS = 5
_ip_issued: dict[str, list[float]] = {}


def anonymous_allowed(ip: str) -> bool:
    """Limit how many demo keys one address can mint per hour."""
    now = time.time()
    with _lock:
        stamps = [t for t in _ip_issued.get(ip, []) if now - t < _IP_WINDOW_SECONDS]
        if len(stamps) >= _IP_MAX_KEYS:
            _ip_issued[ip] = stamps
            return False
        stamps.append(now)
        _ip_issued[ip] = stamps
    return True


def reset_ip_throttle() -> None:
    """Test helper."""
    with _lock:
        _ip_issued.clear()


def load_or_create_bootstrap_key() -> str | None:
    """Return a stable key for local/self-hosted use, from the environment.

    Set ``DR_AISHU_BOOTSTRAP_KEY`` to pin a known key (useful for CI, demos and
    self-hosting). When unset, nothing is created and callers must sign up.
    """
    pinned = os.environ.get("DR_AISHU_BOOTSTRAP_KEY")
    if not pinned:
        return None
    if find_by_plaintext(pinned) is None:
        key_id = secrets.token_hex(8)
        with _lock, store._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO api_keys (id, key_hash, display, tier, email, label,
                       created_at, revoked, request_count)
                   VALUES (?, ?, ?, 'pro', ?, 'bootstrap', ?, 0, 0)""",
                (key_id, hash_key(pinned), _display(pinned), "bootstrap@local", _iso(_now())),
            )
    return pinned
