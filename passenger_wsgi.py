"""Passenger entrypoint for DOM Cloud.

DOM Cloud serves Python apps through Phusion Passenger, which speaks WSGI.
Dr. Aishu is an ASGI (FastAPI) app, so this module adapts one to the other and
exposes the callable Passenger looks for: ``application``.

Everything below the adapter is the same app the Dockerfile runs under uvicorn.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Passenger starts the process from an arbitrary working directory, so make the
# project root importable and keep runtime state inside the deployed tree.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DR_AISHU_DATA_DIR", str(ROOT / "data"))

from a2wsgi import ASGIMiddleware  # noqa: E402

from app.main import app as asgi_app  # noqa: E402

application = ASGIMiddleware(asgi_app)
