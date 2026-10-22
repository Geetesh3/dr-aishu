"""Both ASGI entrypoints must resolve.

Hosting wizards generate a start command without knowing the project layout.
DOM Cloud's Uvicorn preset produced ``uvicorn main:app``, which fails against
this repository because the app lives in ``app/main.py``. A root-level
``main.py`` re-exports it, so both spellings work; these tests keep that true.
"""

from __future__ import annotations

import importlib


def test_root_main_module_exposes_the_app():
    module = importlib.import_module("main")
    assert hasattr(module, "app"), "root main.py must expose `app` for `uvicorn main:app`"


def test_root_main_is_the_same_object_as_app_main():
    root_main = importlib.import_module("main")
    app_main = importlib.import_module("app.main")
    assert root_main.app is app_main.app, "both entrypoints must serve the same app instance"
