"""Root-level ASGI entrypoint.

Hosting wizards (including DOM Cloud) commonly generate a start command of
``uvicorn main:app``, which expects a top-level ``main.py``. The real app lives
in ``app/main.py``; this module re-exports it so both spellings work.
"""

from app.main import app  # noqa: F401
