"""Tests for the deployment configuration and the env vars the app reads.

These guard the free-hosting story: the container must start, every config file
must agree with the others, and the documented environment variables must be the
ones the code actually reads. A mismatch here means a deploy that silently
ignores a setting, which is the hardest kind to notice from the outside.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

DOCKERFILE = ROOT / "Dockerfile"
RENDER = ROOT / "render.yaml"
FLY = ROOT / "fly.toml"
PROCFILE = ROOT / "Procfile"
ENV_EXAMPLE = ROOT / ".env.example"
DOCKERIGNORE = ROOT / ".dockerignore"
REQUIREMENTS = ROOT / "requirements.txt"

# Environment variables the application actually reads.
APP_ENV_VARS = {"DR_AISHU_DATA_DIR", "DR_AISHU_REQUIRE_KEY", "DR_AISHU_BOOTSTRAP_KEY"}


# ---- files exist -------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    [DOCKERFILE, RENDER, FLY, PROCFILE, ENV_EXAMPLE, DOCKERIGNORE, REQUIREMENTS],
    ids=lambda p: p.name,
)
def test_deployment_file_exists(path):
    assert path.is_file(), f"{path.name} is missing"


# ---- Dockerfile --------------------------------------------------------------
def test_dockerfile_installs_tesseract():
    """OCR is a first-class feature, so the image must carry the binary."""
    text = DOCKERFILE.read_text()
    assert "tesseract-ocr" in text


def test_dockerfile_runs_uvicorn_on_the_injected_port():
    text = DOCKERFILE.read_text()
    assert "uvicorn" in text
    assert "app.main:app" in text
    # A hard-coded port would break Render/Fly, which inject PORT.
    assert "${PORT}" in text or "$PORT" in text


def test_dockerfile_copies_every_runtime_directory():
    text = DOCKERFILE.read_text()
    for needed in ("app", "static", "templates"):
        assert re.search(rf"COPY\s+{needed}\b", text), f"Dockerfile does not copy {needed}/"


def test_dockerfile_copies_requirements_before_code():
    """Copying requirements first keeps the dependency layer cached."""
    text = DOCKERFILE.read_text()
    assert text.index("COPY requirements.txt") < text.index("COPY app")


def test_dockerfile_data_dir_matches_app_expectation():
    text = DOCKERFILE.read_text()
    assert "DR_AISHU_DATA_DIR=/data" in text.replace(" ", "")


# ---- .dockerignore -----------------------------------------------------------
def test_dockerignore_excludes_secrets_and_local_data():
    """Match whole ignore lines: a substring check would pass on `.env.*` alone."""
    lines = {
        line.strip()
        for line in DOCKERIGNORE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for pattern in (".env", "*.db", "tests/"):
        assert pattern in lines, f".dockerignore should exclude {pattern}"


def test_dockerignore_does_not_exclude_the_knowledge_base():
    """The app needs app/ and data/models/ to answer at all."""
    text = DOCKERIGNORE.read_text()
    assert not re.search(r"^data/?$", text, re.MULTILINE), "excluding data/ drops the model"
    assert not re.search(r"^app/?$", text, re.MULTILINE)


# ---- render.yaml -------------------------------------------------------------
def test_render_declares_a_docker_web_service_with_a_health_check():
    text = RENDER.read_text()
    assert "type: web" in text
    assert "runtime: docker" in text
    assert "healthCheckPath: /api/health" in text


def test_render_data_dir_matches_dockerfile():
    text = RENDER.read_text()
    assert "/data" in text


def test_render_pins_a_bootstrap_key_that_survives_redeploys():
    """Render's free disk is ephemeral, so a generated key must be pinned."""
    text = RENDER.read_text()
    assert "DR_AISHU_BOOTSTRAP_KEY" in text


# ---- fly.toml ----------------------------------------------------------------
def test_fly_points_at_the_dockerfile():
    text = FLY.read_text()
    assert 'dockerfile = "Dockerfile"' in text


def test_fly_health_check_hits_the_app():
    text = FLY.read_text()
    assert "/api/health" in text


def test_fly_volume_matches_data_dir():
    """A mounted volume is only useful if the app writes to the mount point."""
    text = FLY.read_text()
    assert 'destination = "/data"' in text
    assert "DR_AISHU_DATA_DIR" in text and "/data" in text


def test_fly_internal_port_matches_env_port():
    text = FLY.read_text()
    internal = re.search(r"internal_port\s*=\s*(\d+)", text)
    env_port = re.search(r'PORT\s*=\s*"(\d+)"', text)
    assert internal and env_port
    assert internal.group(1) == env_port.group(1)


# ---- Procfile ----------------------------------------------------------------
def test_procfile_uses_the_injected_port():
    text = PROCFILE.read_text()
    assert "uvicorn" in text and "app.main:app" in text
    assert "$PORT" in text


# ---- environment contract ----------------------------------------------------
def test_env_example_documents_every_variable_the_app_reads():
    text = ENV_EXAMPLE.read_text()
    documented = {m for m in APP_ENV_VARS if m in text}
    assert documented == APP_ENV_VARS, f"undocumented: {APP_ENV_VARS - documented}"


def test_env_example_does_not_invent_variables():
    """Every DR_AISHU_* name in the template must be one the app reads."""
    text = ENV_EXAMPLE.read_text()
    mentioned = set(re.findall(r"DR_AISHU_[A-Z_]+", text))
    assert mentioned <= APP_ENV_VARS, f"not read by the app: {mentioned - APP_ENV_VARS}"


def test_env_example_keeps_the_bootstrap_key_commented_out():
    """A committed default key would be a shared secret in every deployment."""
    for line in ENV_EXAMPLE.read_text().splitlines():
        if "DR_AISHU_BOOTSTRAP_KEY=" in line:
            assert line.lstrip().startswith("#"), "bootstrap key must not be active by default"


def test_requirements_include_every_runtime_import():
    """Compare package names, not raw substrings, so a typo cannot pass."""
    declared = set()
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!\[; ]", line, maxsplit=1)[0].strip().lower()
        if name:
            declared.add(name)
    for package in ("fastapi", "uvicorn", "jinja2", "python-multipart", "pillow"):
        assert package in declared, f"{package} missing from requirements.txt"

# ---- front-end contracts -----------------------------------------------------
STATIC = ROOT / "static"
NOTEBOOK_BUILDER = ROOT / "scripts" / "build_deploy_notebook.py"


def test_all_pages_use_the_same_api_key_slot():
    """Regression: insights.js read a different storage slot than app.js.

    developers.js writes the key to sessionStorage["draishu_dev_key"]. A page
    that reads a different key or storage silently fails auth, and because the
    failure is a rejected fetch the page just looks empty.
    """
    pages = {
        name: (STATIC / name).read_text()
        for name in ("app.js", "insights.js", "developers.js")
    }

    for name, source in pages.items():
        assert "draishu_dev_key" in source, f"{name} must use the shared key slot"
        assert "sessionStorage" in source, f"{name} must read sessionStorage"

    assert "dr_aishu_key" not in pages["insights.js"]


def test_notebook_builder_uses_real_routine_block_fields():
    """Regression: the deploy notebook printed block["activity"], which is a
    category value rather than a key, so the cell raised KeyError."""
    source = NOTEBOOK_BUILDER.read_text()
    assert 'block["activity"]' not in source
    assert 'block["title"]' in source

    notebook = json.loads((ROOT / "notebooks" / "Dr_Aishu_Deploy.ipynb").read_text())
    for cell in notebook["cells"]:
        assert 'block["activity"]' not in "".join(cell["source"])


# ---- operator-only surfaces --------------------------------------------------
DATASET_TEMPLATE = ROOT / "templates" / "dataset.html"
ASSESS_TEMPLATE = ROOT / "templates" / "assess.html"
DATASET_JS = STATIC / "dataset.js"


def test_assess_page_has_no_learning_or_training_ui():
    """Regression: the learning panel let any visitor retrain and export the
    model. The UI is gone; the endpoints behind it are operator-only now."""
    source = ASSESS_TEMPLATE.read_text()
    for marker in ("learning", "retrain", "loadLearningStats"):
        assert marker not in source, f"assess.html still references {marker!r}"

    app_js = (STATIC / "app.js").read_text()
    for marker in ("loadLearningStats", "learning-panel", "/api/model/train"):
        assert marker not in app_js, f"app.js still references {marker!r}"


def test_dataset_page_is_gated_and_absent_from_nav():
    """Regression: /dataset was linked in the public nav and rendered the
    operator payload to anyone who opened it."""
    base = (ROOT / "templates" / "base.html").read_text()
    assert 'href="/dataset"' not in base, "dataset link must not be in the nav"

    source = DATASET_TEMPLATE.read_text()
    assert 'id="ds-gate"' in source, "the operator gate must be present"
    assert 'id="ds-content" hidden' in source, "gated content must start hidden"

    js = DATASET_JS.read_text()
    assert "draishu_dev_key" in js, "the gate must reuse the shared key slot"
    assert "operator" in js.lower(), "the gate must distinguish operator access"


def test_dataset_css_hides_the_gate_when_unlocked():
    """Regression: `.gate { display: grid }` outranks the UA [hidden] rule, so
    the gate stayed on screen after a successful unlock."""
    css = (STATIC / "dataset.css").read_text()
    assert ".gate[hidden]" in css, "the gate needs an explicit [hidden] rule"

