"""Shared test fixtures.

Two pieces of module-level state have to be reset between tests:

* the learning module holds the active model, and a model trained by one test
  would otherwise leak into the next and change pipeline results;
* the security module caches one anonymous key per client address, which would
  otherwise let a rate-limit test bleed into unrelated tests.
"""
from __future__ import annotations

import pytest

from app import auth, image_model, learning, security, store


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point the store and the auth tables at a throwaway database.

    The auth module caches which databases it has created tables in, so that
    cache is cleared too; otherwise a second test would reuse the first test's
    schema marker against a brand-new file.
    """
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "UPLOAD_DIR", tmp_path / "uploads")
    store.init_db()
    auth._ready.clear()
    auth.init_db()
    yield


@pytest.fixture(autouse=True)
def reset_learning_state():
    """Snapshot and restore the active model around every test."""
    saved = dict(learning._state)
    learning._state.update({"backend": None, "model": None, "meta": {}})
    yield
    learning._state.clear()
    learning._state.update(saved)


@pytest.fixture(autouse=True)
def reset_image_model():
    """Drop any loaded image model so a monkeypatched one cannot leak out."""
    image_model.reset()
    yield
    image_model.reset()


@pytest.fixture(autouse=True)
def reset_auth_state():
    """Clear cached anonymous keys and per-IP issuance throttling."""
    security.reset_anonymous_cache()
    auth.reset_ip_throttle()
    yield
    security.reset_anonymous_cache()
    auth.reset_ip_throttle()


@pytest.fixture
def api_key(isolated_db):
    """Plaintext of a free-tier key backed by the test database."""
    plaintext, _key = auth.create_key(tier="free", email="tester@example.com", label="pytest")
    return plaintext


@pytest.fixture
def api_headers(api_key):
    return {"X-API-Key": api_key}


@pytest.fixture
def operator_key(isolated_db):
    """Plaintext of a Pro (operator) key.

    The Pro tier is the only one carrying the export and train scopes, and it is
    not self-service, so tests that need those scopes mint one directly rather
    than going through the signup endpoint.
    """
    plaintext, _key = auth.create_key(tier="pro", email="operator@example.com", label="pytest-op")
    return plaintext


@pytest.fixture
def operator_headers(operator_key):
    return {"X-API-Key": operator_key}
