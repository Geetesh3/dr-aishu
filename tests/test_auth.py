"""Tests for API key issuance, authentication, scopes and rate limiting.

The security properties that matter here are: a key's plaintext is never
recoverable from storage, an invalid or revoked key is rejected, a tier cannot
reach a scope it was not granted, one tenant cannot touch another tenant's keys,
and the rate limiter actually stops a caller.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import auth, security
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ---- key issuance -----------------------------------------------------------
def test_key_plaintext_is_never_stored():
    plaintext, key = auth.create_key(tier="free", email="a@example.com")
    with auth.store._connect() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key.id,)).fetchone()
    assert plaintext not in dict(row).values()
    assert row["key_hash"] == auth.hash_key(plaintext)


def test_signup_returns_secret_once(client):
    res = client.post("/api/keys/signup", json={"email": "dev@example.com", "label": "proj"})
    assert res.status_code == 201
    body = res.json()
    assert body["secret"].startswith(auth.KEY_PREFIX)
    assert body["tier"] == "free"
    assert "not recoverable" in body["warning"]

    # Listing the keys again must not reveal the secret.
    listed = client.get(
        "/api/keys/list", params={"email": "dev@example.com"}, headers={"X-API-Key": body["secret"]}
    ).json()["keys"]
    assert len(listed) == 1
    assert "secret" not in listed[0]
    assert body["secret"] not in listed[0].values()


def test_signup_rejects_malformed_email(client):
    assert client.post("/api/keys/signup", json={"email": "not-an-email"}).status_code == 422


def test_pro_tier_is_not_self_service(client):
    """Pro carries the operator scopes (export, train) and must not be mintable.

    Regression: signup accepted tier="pro" from any caller, so a visitor could
    issue themselves a key that exports every consented case.
    """
    for body in ({"tier": "pro"}, {"tier": "pro", "email": "pro@example.com"}):
        assert client.post("/api/keys/signup", json=body).status_code == 403


def test_signup_is_throttled_per_address(client):
    for _ in range(auth._IP_MAX_KEYS):
        assert client.post("/api/keys/signup", json={}).status_code == 201
    assert client.post("/api/keys/signup", json={}).status_code == 429


# ---- authentication ---------------------------------------------------------
def test_analyze_works_without_a_key_as_demo(client):
    """The hosted demo must be usable straight from the browser."""
    res = client.post("/api/analyze", json={"text": "fever and cough for 2 days"})
    assert res.status_code == 200
    assert res.json()["diagnoses"]


def test_valid_key_authenticates(client, api_key):
    res = client.post(
        "/api/analyze", json={"text": "fever for 2 days"}, headers={"X-API-Key": api_key}
    )
    assert res.status_code == 200


def test_bearer_header_also_works(client, api_key):
    res = client.post(
        "/api/analyze",
        json={"text": "fever for 2 days"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert res.status_code == 200


def test_invalid_key_is_rejected(client):
    res = client.post(
        "/api/analyze", json={"text": "fever"}, headers={"X-API-Key": "dra_totally_bogus"}
    )
    assert res.status_code == 401
    assert "Invalid API key" in res.json()["detail"]


def test_revoked_key_is_rejected(client, api_key):
    me = client.get("/api/keys/me", headers={"X-API-Key": api_key}).json()
    assert client.post(
        "/api/keys/revoke", json={"key_id": me["id"]}, headers={"X-API-Key": api_key}
    ).status_code == 200
    res = client.post("/api/analyze", json={"text": "fever"}, headers={"X-API-Key": api_key})
    assert res.status_code == 401
    assert "revoked" in res.json()["detail"].lower()


def test_require_key_mode_blocks_anonymous(monkeypatch, client):
    monkeypatch.setattr(security, "REQUIRE_KEY", True)
    assert client.post("/api/analyze", json={"text": "fever"}).status_code == 401


# ---- scopes -----------------------------------------------------------------
def test_free_key_cannot_train(client, api_key):
    res = client.post("/api/model/train", json={}, headers={"X-API-Key": api_key})
    assert res.status_code == 403
    assert "train" in res.json()["detail"]


def test_anonymous_key_cannot_export(client):
    assert client.get("/api/export/training-data").status_code == 403


def test_pro_key_can_reach_train_scope(client):
    plaintext, _key = auth.create_key(tier="pro", email="pro@example.com")
    res = client.post("/api/model/train", json={}, headers={"X-API-Key": plaintext})
    # Scope is satisfied, so the failure (if any) is not an auth failure.
    assert res.status_code != 403


def test_keys_me_reports_scopes_and_usage(client, api_key):
    body = client.get("/api/keys/me", headers={"X-API-Key": api_key}).json()
    assert body["tier"] == "free"
    assert "analyze" in body["scopes"]
    assert "train" not in body["scopes"]
    assert body["anonymous"] is False
    assert body["usage"]["minute"]["limit"] == auth.TIERS["free"]["per_minute"]


def test_anonymous_caller_can_read_its_own_identity(client):
    """The web UI must learn its scopes without holding a key.

    Regression: ``/api/keys/me`` required the ``keys`` scope, which the
    anonymous tier does not grant, so the browser UI received a 403 and left
    Pro-only buttons visible for visitors who could not use them.
    """
    res = client.get("/api/keys/me")
    assert res.status_code == 200
    body = res.json()
    assert body["anonymous"] is True
    assert body["tier"] == "anonymous"
    assert "analyze" in body["scopes"]
    assert "train" not in body["scopes"]


def test_anonymous_identity_does_not_leak_internal_record(client):
    """The per-IP key id and label are internal and must not be echoed back."""
    body = client.get("/api/keys/me").json()
    assert body["id"] is None
    assert body["label"] is None
    assert "auto:" not in json.dumps(body)


def test_identity_endpoint_still_rejects_a_bogus_key(client):
    res = client.get("/api/keys/me", headers={"X-API-Key": "dra_totally_bogus"})
    assert res.status_code == 401


# ---- tenant isolation -------------------------------------------------------
def test_key_cannot_list_another_owners_keys(client):
    plaintext, _ = auth.create_key(tier="free", email="owner@example.com")
    res = client.get(
        "/api/keys/list", params={"email": "victim@example.com"}, headers={"X-API-Key": plaintext}
    )
    assert res.status_code == 403


def test_key_cannot_revoke_another_owners_key(client):
    plaintext, _ = auth.create_key(tier="free", email="owner@example.com")
    _other_plain, victim = auth.create_key(tier="free", email="victim@example.com")
    res = client.post(
        "/api/keys/revoke",
        json={"key_id": victim.id, "email": "victim@example.com"},
        headers={"X-API-Key": plaintext},
    )
    assert res.status_code == 403
    assert not auth.get_key(victim.id).revoked


def test_revoke_requires_email_for_other_keys(client, api_key):
    _other_plain, victim = auth.create_key(tier="free", email="victim@example.com")
    res = client.post(
        "/api/keys/revoke", json={"key_id": victim.id}, headers={"X-API-Key": api_key}
    )
    assert res.status_code == 422


# ---- deployment internals are operator-only ---------------------------------
def test_free_tier_cannot_export_or_train(client, api_key):
    """The scopes that read stored cases and retrain must not come free.

    Regression: the free tier carried the export scope, so anyone who signed up
    with any unverified email could download every consented narrative.
    """
    headers = {"X-API-Key": api_key}
    assert client.get("/api/export/training-data", headers=headers).status_code == 403
    assert client.post("/api/model/train", json={}, headers=headers).status_code == 403


def test_dataset_endpoints_are_operator_only(client, api_key):
    """The dataset browser exposes build internals, so it is not public."""
    for path in (
        "/api/dataset/status",
        "/api/dataset/diseases",
        "/api/dataset/images",
        "/api/dataset/sources",
        "/api/dataset/medicinenet",
    ):
        assert client.get(path).status_code == 403, path
        assert client.get(path, headers={"X-API-Key": api_key}).status_code == 403, path


def test_dataset_status_does_not_leak_server_paths(client, operator_headers):
    """Internal file locations are stripped even for the operator."""
    body = client.get("/api/dataset/status", headers=operator_headers).json()
    assert "path" not in body
    assert "path" not in body.get("image_model", {})


def test_model_status_does_not_leak_server_paths(client):
    body = client.get("/api/model/status").json()
    assert "db_path" not in body["data"]
    assert "path" not in body.get("image_model", {})


# ---- rate limiting ----------------------------------------------------------
def test_rate_limit_blocks_and_reports_retry_after(client):
    plaintext, _key = auth.create_key(tier="anonymous")
    limit = auth.TIERS["anonymous"]["per_minute"]

    for _ in range(limit):
        assert client.post(
            "/api/analyze", json={"text": "fever"}, headers={"X-API-Key": plaintext}
        ).status_code == 200

    res = client.post("/api/analyze", json={"text": "fever"}, headers={"X-API-Key": plaintext})
    assert res.status_code == 429
    assert res.headers["Retry-After"] == "60"


def test_rate_limit_headers_present_on_success(client, api_key):
    res = client.post(
        "/api/analyze", json={"text": "fever for 2 days"}, headers={"X-API-Key": api_key}
    )
    assert res.headers["X-RateLimit-Limit-Minute"] == str(auth.TIERS["free"]["per_minute"])
    assert int(res.headers["X-RateLimit-Remaining-Minute"]) == auth.TIERS["free"]["per_minute"] - 1
    assert res.headers["X-RateLimit-Tier"] == "free"


def test_blocked_request_does_not_consume_daily_quota():
    """A throttled call must not burn the caller's remaining daily budget."""
    _plaintext, key = auth.create_key(tier="anonymous")
    limit = auth.TIERS["anonymous"]["per_minute"]
    for _ in range(limit):
        auth.consume(key)

    allowed, _ = auth.consume(key)
    assert allowed is False
    assert auth.usage(key.id)["day"]["used"] == limit


# ---- bootstrap key ----------------------------------------------------------
def test_bootstrap_key_is_created_from_env(monkeypatch):
    monkeypatch.setenv("DR_AISHU_BOOTSTRAP_KEY", "dra_pinned_selfhost_key")
    assert auth.load_or_create_bootstrap_key() == "dra_pinned_selfhost_key"
    found = auth.find_by_plaintext("dra_pinned_selfhost_key")
    assert found is not None and found.tier == "pro"
    # Idempotent: calling twice must not raise or duplicate.
    assert auth.load_or_create_bootstrap_key() == "dra_pinned_selfhost_key"


def test_no_bootstrap_key_without_env(monkeypatch):
    monkeypatch.delenv("DR_AISHU_BOOTSTRAP_KEY", raising=False)
    assert auth.load_or_create_bootstrap_key() is None


# ---- developer page ---------------------------------------------------------
def test_developers_page_renders(client):
    res = client.get("/developers")
    assert res.status_code == 200
    assert "API" in res.text


def test_developers_page_documents_real_response_fields(client):
    """The docs must name fields the API actually returns.

    Regression: the page advertised ``medicines.schedule`` and
    ``routine.blocks[].activity``, neither of which exists. Copy-pasted sample
    code that reads them raises KeyError, so developers would hit a failure on
    their first call and blame the API.
    """
    page = client.get("/developers").text
    analysis = client.post("/api/analyze", json={"text": "sore throat and fever"}).json()

    for field in ("timetable",):
        assert field in analysis["medicines"]
        assert f'"{field}"' in page, f"{field} should appear in the documented response shape"

    for wrong in ('"schedule"', 'block["activity"]'):
        assert wrong not in page, f"documented field {wrong} does not exist in the API"

    block = analysis["routine"]["blocks"][0]
    for key in ("time", "title"):
        assert key in block, f"quickstart prints block[{key!r}] but the API omits it"

    # The per-problem plan is documented, so its fields must be the real ones.
    assert "solutions" in analysis
    assert analysis["solutions"]["problems"], "expected a per-problem plan"
    for key in ("problems", "condition", "notes"):
        assert key in analysis["solutions"]
        assert f'"{key}"' in page, f"solutions.{key} should be documented"


def test_tiers_endpoint_is_public(client):
    body = client.get("/api/keys/tiers").json()
    ids = {t["id"] for t in body["tiers"]}
    assert ids == {"free", "pro"}
    assert body["auth_header"] == "X-API-Key"


# ---- storage pragmas ---------------------------------------------------------
def test_connect_uses_the_wal_durability_pairing():
    """WAL without synchronous=NORMAL fsyncs every commit; recording an
    analysis then dominates the request. Pin the pairing so a future edit does
    not silently reintroduce the slow setting."""
    from app import store

    conn = store._connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    finally:
        conn.close()
