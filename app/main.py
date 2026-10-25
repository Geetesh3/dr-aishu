"""Dr. Aishu — FastAPI application.

Exposes symptom analysis (text and image), severity triage, medicine suggestions,
daily routines, plus the data-collection and model-training endpoints that let the
assistant improve from consented real cases.
"""
from __future__ import annotations

import hashlib
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from . import (
    analytics, auth, conditions, dataset, demo_data, drugref, i18n, image_model,
    learning, pipeline, security, store, vision, worlddata,
)
from .kb import KB
from .translations import bundle as translation_bundle, catalogue as translation_catalogue

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# Static assets are served with a long cache lifetime. The browser then reuses
# them across page navigations instead of re-downloading the 21 kB stylesheet
# and 31 kB of JavaScript on every click.
STATIC_CACHE_SECONDS = 60 * 60 * 24 * 30

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Ensure the auth tables exist and honour a pinned bootstrap key."""
    auth.init_db()
    if auth.load_or_create_bootstrap_key():
        print("[dr-aishu] bootstrap API key loaded from DR_AISHU_BOOTSTRAP_KEY")

    # Compile the matcher buckets off the request path. The first request for a
    # script otherwise pays ~0.5 s of regex compilation; doing it here, in a
    # thread, means the process is ready before a user is. It is a daemon thread
    # and best-effort: a request that arrives first simply compiles the bucket
    # itself, which is the behaviour without this call.
    threading.Thread(target=KB.prewarm, daemon=True).start()
    yield


app = FastAPI(
    title="Dr. Aishu",
    description=(
        "Medical AI: symptom analysis, severity triage, medicine timetable and daily "
        "routines. Authenticate with an API key from /developers using the "
        "`X-API-Key` header, or `Authorization: Bearer <key>`."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

class CachedStaticFiles(StaticFiles):
    """StaticFiles that tells the browser how long it may reuse each asset.

    Without a Cache-Control header the browser revalidates on every navigation,
    which costs a round trip per asset before the page can paint. The ETag is
    kept so a stale copy still revalidates correctly after the cache expires.
    """

    def file_response(self, *args: Any, **kwargs: Any):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = f"public, max-age={STATIC_CACHE_SECONDS}"
        return response


app.mount("/static", CachedStaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def _asset_version() -> str:
    """A short token that changes whenever any static asset changes.

    Assets are served with a 30-day max-age, so without this a browser keeps
    running the previous app.js after a deploy. Hashing the file mtimes is
    cheap (a stat per file, done once at import) and makes /static/app.js?v=…
    a new URL, which is what actually forces the refresh.
    """
    stamps = []
    for path in sorted(STATIC_DIR.rglob("*")):
        if path.is_file():
            stamps.append(f"{path.name}:{path.stat().st_mtime_ns}")
    digest = hashlib.sha1("|".join(stamps).encode("utf-8")).hexdigest()
    return digest[:10]


ASSET_VERSION = _asset_version()
templates.env.globals["asset_version"] = ASSET_VERSION

# Compress every response over 1 kB. The templates, the CSS/JS bundles and the
# 254k-row product search payloads all shrink by 70-85%, which is most of the
# perceived load time on a slow mobile connection.
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Replays rate-limit headers onto responses built inside endpoints.
app.middleware("http")(security.rate_limit_middleware)

MAX_IMAGE_BYTES = 12 * 1024 * 1024
# Content types are a hint, not a gate: phones send `application/octet-stream`
# or `image/heic` for formats Pillow reads fine. The bytes are sniffed in
# `vision.is_supported` and anything decodable is accepted.
ACCEPTED_IMAGE_HINT = (
    "image/*, application/pdf, application/octet-stream"
)

# Load any previously trained model at import time.
ACTIVE_BACKEND = learning.load_model()


# ---- cached, immutable views of the language catalogue ----------------------
# Both are computed once on first use and then reused. They describe static
# data (the shipped translation tables), so there is nothing to invalidate
# short of a redeploy.
_catalogue_cache: dict[str, dict[str, Any]] | None = None
_all_bundles_cache: dict[str, dict[str, str]] | None = None


def _catalogue() -> dict[str, dict[str, Any]]:
    """Coverage for every language, computed at most once per process."""
    global _catalogue_cache
    if _catalogue_cache is None:
        _catalogue_cache = translation_catalogue()
    return _catalogue_cache


def _all_bundles() -> dict[str, dict[str, str]]:
    """Resolved string tables for every language, computed at most once."""
    global _all_bundles_cache
    if _all_bundles_cache is None:
        codes = [e["code"] for e in i18n.language_catalogue()] + ["en"]
        _all_bundles_cache = {code: translation_bundle(code) for code in codes}
    return _all_bundles_cache


# ---- request models ---------------------------------------------------------
class AnalyzeRequest(BaseModel):
    text: str = ""
    top_n: int = Field(6, ge=1, le=12)
    include_rx: bool = True
    age: float | None = Field(None, ge=0, le=120)
    weight_kg: float | None = Field(None, gt=0, le=400)
    pregnant: bool = False
    chronic_conditions: str | None = None
    allergies: str | None = None
    current_medicines: str | None = None
    sex: str | None = None
    consent_share: bool = True
    lang: str = "auto"


class FeedbackRequest(BaseModel):
    case_id: str
    correct_diagnosis: str | None = None
    is_correct: bool | None = None
    rating: int | None = Field(None, ge=1, le=5)
    comment: str | None = None


class TrainRequest(BaseModel):
    backend: str = Field("tfidf", pattern="^(tfidf|transformer)$")
    min_confidence: float = Field(0.0, ge=0, le=100)
    use_seed: bool = True
    epochs: int = Field(4, ge=1, le=20)


class KeySignupRequest(BaseModel):
    email: str | None = Field(None, max_length=200)
    label: str | None = Field(None, max_length=100)
    tier: str = Field("free", pattern="^(free|pro)$")


class KeyRevokeRequest(BaseModel):
    key_id: str
    email: str | None = Field(None, max_length=200)


_EMAIL_RE = __import__("re").compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def _valid_email(value: str | None) -> str | None:
    """Minimal email check; avoids a hard dependency on email-validator."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if not _EMAIL_RE.match(value):
        raise HTTPException(status_code=422, detail="That does not look like a valid email address.")
    return value.lower()


def _page(request: Request, template: str, active: str) -> HTMLResponse:
    """Render a page with the language context the base template needs.

    The interface language comes from the `?lang=` query parameter, falling back
    to English. It is carried in the URL rather than a cookie so a shared link
    opens in the same language for the recipient.
    """
    lang = request.query_params.get("lang") or "en"
    if lang != "en" and not i18n.is_supported(lang):
        lang = "en"
    meta = i18n.LANGUAGE_BY_CODE.get(lang)
    return templates.TemplateResponse(request, template, {
        "active": active,
        "lang": lang,
        "rtl": bool(meta and meta.rtl),
    })


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return _page(request, "home.html", "home")


@app.get("/assess", response_class=HTMLResponse)
async def assess_page(request: Request) -> HTMLResponse:
    return _page(request, "assess.html", "assess")


@app.get("/library", response_class=HTMLResponse)
async def library_page(request: Request) -> HTMLResponse:
    return _page(request, "library.html", "library")


@app.get("/medicines", response_class=HTMLResponse)
async def medicines_page(request: Request) -> HTMLResponse:
    return _page(request, "medicines.html", "medicines")


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request) -> HTMLResponse:
    return _page(request, "about.html", "about")


@app.get("/developers", response_class=HTMLResponse)
async def developers(request: Request) -> HTMLResponse:
    """Self-service page for issuing and managing API keys. Free, no card."""
    return _page(request, "developers.html", "developers")


@app.get("/insights", response_class=HTMLResponse)
async def insights(request: Request) -> HTMLResponse:
    """Public dashboard over recorded searches: what people ask, how severe."""
    return _page(request, "insights.html", "insights")


@app.get("/dataset", response_class=HTMLResponse)
async def dataset_page(request: Request) -> HTMLResponse:
    """Browse the consolidated external dataset (the built Excel workbook)."""
    return _page(request, "dataset.html", "dataset")


# ---- consolidated dataset ---------------------------------------------------
# The dataset browser exposes build internals: internal file paths, archive
# hashes, exact per-class image counts and the training composition of the
# shipped model. None of that is needed to use Dr. Aishu, and all of it helps
# someone reconstruct the deployment, so every endpoint here is operator-only.
@app.get("/api/dataset/status")
async def dataset_status(
    ctx: security.AuthContext = Depends(security.require_operator),
) -> dict[str, Any]:
    """Which sheets the workbook has and how many rows each holds."""
    return {**_public_dataset_status(), "image_model": _public_image_model()}


@app.get("/api/dataset/diseases")
async def dataset_diseases(
    in_kb: bool = False,
    limit: int = 50,
    offset: int = 0,
    ctx: security.AuthContext = Depends(security.require_operator),
) -> dict[str, Any]:
    rows = dataset.diseases(in_kb_only=in_kb)
    total = len(rows)
    window = rows[offset: offset + limit] if limit > 0 else rows[offset:]
    return {"total": total, "offset": offset, "limit": limit, "items": window}


@app.get("/api/dataset/diseases/{name}")
async def dataset_disease(
    name: str,
    ctx: security.AuthContext = Depends(security.require_operator),
) -> JSONResponse:
    row = dataset.find_disease(name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no condition matching {name!r}")
    return JSONResponse(row)


@app.get("/api/dataset/images")
async def dataset_images(
    modality: str | None = None,
    limit: int = 50,
    ctx: security.AuthContext = Depends(security.require_operator),
) -> dict[str, Any]:
    rows = dataset.images(modality=modality)
    return {
        "total": len(rows),
        "labels": dataset.image_label_counts(),
        "items": rows[:limit] if limit > 0 else rows,
    }


@app.get("/api/dataset/sources")
async def dataset_sources(
    ctx: security.AuthContext = Depends(security.require_operator),
) -> dict[str, Any]:
    """Provenance for every source, including its licence and SHA-256."""
    return {"items": dataset.sources()}


@app.get("/api/dataset/medicinenet")
async def dataset_medicinenet(
    limit: int = 100,
    letter: str | None = None,
    ctx: security.AuthContext = Depends(security.require_operator),
) -> dict[str, Any]:
    rows = dataset.medicinenet_conditions()
    if letter:
        rows = [r for r in rows if (r.get("letter") or "").upper() == letter.upper()]
    return {"total": len(rows), "items": rows[:limit] if limit > 0 else rows}


@app.get("/api/dataset/deeplesion")
async def dataset_deeplesion(
    lesion_type: str | None = None,
    limit: int = 50,
    ctx: security.AuthContext = Depends(security.require_operator),
) -> dict[str, Any]:
    """DeepLesion lesion annotations (radiology ground truth, not symptom data)."""
    rows = dataset.sheet("DeepLesion")
    if lesion_type:
        rows = [r for r in rows if str(r.get("coarse_lesion_type")) == lesion_type]
    return {
        "total": len(rows),
        "lesion_types": dataset.lesion_type_counts(),
        "items": rows[:limit] if limit > 0 else rows,
    }


@app.get("/api/dataset/catalogue")
async def dataset_catalogue(
    ctx: security.AuthContext = Depends(security.require_operator),
) -> dict[str, Any]:
    """Public disease datasets catalogued by the CC BY review (PMC8950225)."""
    rows = dataset.sheet("Dataset_Catalogue")
    return {"total": len(rows), "items": rows}


# ---- API key management -----------------------------------------------------
@app.post("/api/keys/signup")
async def keys_signup(payload: KeySignupRequest, request: Request) -> JSONResponse:
    """Issue a free API key. The plaintext key is returned exactly once.

    Only the free tier is self-service. The Pro tier carries the operator scopes
    (data export and retraining), so it is issued by the operator out of band --
    otherwise any visitor could mint a key that reads every stored case.
    """
    email = _valid_email(payload.email)
    tier = payload.tier
    if tier != "free":
        raise HTTPException(
            status_code=403,
            detail="Only the free tier is self-service. The Pro tier is issued "
                   "by the operator; see /developers for the free tier.",
        )

    client_ip = security._client_ip(request)
    if not auth.anonymous_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many keys requested from this address. Try again later.",
        )

    plaintext, key = auth.create_key(
        tier=tier, email=email, label=payload.label, ip=client_ip
    )
    body = key.to_public()
    body["secret"] = plaintext
    body["warning"] = "Store this key now. It is not recoverable and will not be shown again."
    return JSONResponse(body, status_code=201)


@app.get("/api/keys/me")
async def keys_me(ctx: security.AuthContext = Depends(security.require_identity)) -> dict[str, Any]:
    """Return the calling key's identity, scopes and live usage counters.

    Uses the identity dependency rather than ``require_keys`` so that any caller
    can read its own scopes, including the browser's anonymous key. Without this
    the web UI could not tell which actions to enable for the current visitor.
    """
    body = ctx.key.to_public()
    body["usage"] = auth.usage(ctx.key.id)
    body["anonymous"] = ctx.anonymous
    if ctx.anonymous:
        # The anonymous key is an internal, per-IP record. Its id and label
        # (which encodes the client IP) are not the visitor's to see.
        body["id"] = None
        body["label"] = None
        body["request_count"] = None
    return body


@app.get("/api/keys/list")
async def keys_list(
    email: str,
    ctx: security.AuthContext = Depends(security.require_keys),
) -> dict[str, Any]:
    """List every key issued to an email address.

    The caller must either own that email or be on a tier whose key carries no
    email (self-hosted bootstrap keys), so one tenant cannot enumerate another's
    keys. Plaintext secrets are never returned.
    """
    wanted = (email or "").strip().lower()
    if not wanted:
        raise HTTPException(status_code=422, detail="email is required")
    owner = (ctx.key.email or "").lower()
    if owner and owner != wanted:
        raise HTTPException(status_code=403, detail="This key cannot list another owner's keys.")
    return {"email": wanted, "keys": auth.list_keys(wanted)}


@app.post("/api/keys/revoke")
async def keys_revoke(
    payload: KeyRevokeRequest,
    ctx: security.AuthContext = Depends(security.require_keys),
) -> dict[str, Any]:
    """Revoke a key. Self-revocation is always allowed; revoking another owner's
    key requires knowing their email and being on that key's tier."""
    owner = (ctx.key.email or "").lower()
    email = (payload.email or "").strip().lower() or None
    if payload.key_id == ctx.key.id:
        email = ctx.key.email
    elif email is None:
        raise HTTPException(status_code=422, detail="email is required to revoke another key.")
    elif owner and owner != email:
        raise HTTPException(status_code=403, detail="This key cannot revoke another owner's keys.")

    if not auth.revoke(payload.key_id, email=email):
        raise HTTPException(status_code=404, detail="No matching key found.")
    return {"revoked": payload.key_id}


@app.get("/api/keys/tiers")
async def keys_tiers() -> dict[str, Any]:
    """Public catalogue of tiers, scopes and limits."""
    return {
        "tiers": [
            {
                "id": tid,
                "label": cfg["label"],
                "description": cfg["description"],
                "per_minute": cfg["per_minute"],
                "per_day": cfg["per_day"],
                "scopes": sorted(cfg["scopes"]),
            }
            for tid, cfg in auth.TIERS.items()
            if tid != "anonymous"
        ],
        "auth_header": "X-API-Key",
        "alt_header": "Authorization: Bearer <key>",
        "require_key": security.REQUIRE_KEY,
    }


@app.get("/api/keys/stats")
async def keys_stats(ctx: security.AuthContext = Depends(security.require_keys)) -> dict[str, Any]:
    return auth.stats()


@app.get("/api/languages")
async def languages() -> dict[str, Any]:
    """Every supported language, with translation coverage.

    `safety_complete` tells the client whether the emergency strings exist in
    that language, which is what decides if it is safe to switch the interface.
    """
    coverage = _catalogue()
    out = []
    for entry in i18n.language_catalogue():
        cov = coverage.get(entry["code"], {})
        out.append({
            **entry,
            "translated_percent": cov.get("percent", 0.0),
            "safety_complete": cov.get("safety_complete", False),
            "complete": cov.get("complete", False),
        })
    out.append({
        "code": "en", "english": "English", "native": "English",
        "script": "latin", "rtl": False, "translated_percent": 100.0,
        "safety_complete": True, "complete": True,
    })
    return {"count": len(out), "languages": sorted(out, key=lambda x: x["english"])}


@app.get("/api/languages/all")
async def all_language_bundles() -> JSONResponse:
    """Every language's string table in one response.

    The browser fetches this once and then switches language locally, with no
    further round trip. All 23 tables are ~28 kB gzipped, which is less than a
    single hero image and far cheaper than a request per switch -- on a slow
    connection the per-switch request was the visible delay.
    """
    return JSONResponse(
        {"bundles": _all_bundles()},
        headers={"Cache-Control": f"public, max-age={STATIC_CACHE_SECONDS}"},
    )


@app.get("/api/languages/{code}")
async def language_bundle(code: str) -> JSONResponse:
    """The resolved string table for one language, for the browser to apply."""
    if code != "en" and not i18n.is_supported(code):
        raise HTTPException(status_code=404, detail=f"Unsupported language: {code}")
    meta = i18n.LANGUAGE_BY_CODE.get(code)
    return JSONResponse({
        "code": code,
        "rtl": bool(meta and meta.rtl),
        "native": meta.native if meta else "English",
        "coverage": _catalogue().get(code, {}),
        "strings": translation_bundle(code),
    })


@app.post("/api/detect-language")
async def detect_language(payload: dict[str, Any]) -> dict[str, Any]:
    """Identify the language of a piece of text, without running an assessment.

    Used by the interface to follow the user's language as they type.
    """
    text = str(payload.get("text") or "")
    return i18n.detect_language(text)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "assistant": "Dr. Aishu",
        "version": app.version,
        "knowledge_base": {
            "diseases": len(KB.diseases),
            "symptoms": len(KB.symptoms),
            "medicines": len(KB.medicines),
            "routines": len(KB.routines),
        },
        "model": learning.status(),
        "auth": {
            "key_required": security.REQUIRE_KEY,
            "header": "X-API-Key",
            "signup": "/developers",
        },
    }


@app.get("/api/symptoms")
async def symptoms() -> dict[str, Any]:
    return {
        "symptoms": [
            {
                "id": s.id,
                "label": s.label,
                "system": s.system,
                "red_flag": s.red_flag,
                "red_flag_level": s.red_flag_level,
                "synonyms": s.synonyms[:6],
            }
            for s in sorted(KB.symptoms.values(), key=lambda x: x.label)
        ]
    }


@app.get("/api/diseases")
async def diseases() -> dict[str, Any]:
    return {
        "diseases": [
            {
                "id": d.id,
                "name": d.name,
                "category": d.category,
                "icd10": d.icd10,
                "description": d.description,
                "severity_profile": d.severity_profile,
                "contagious": d.contagious,
                "typical_duration_days": d.typical_duration_days,
                "symptoms": [KB.symptom_label(s) for s in d.symptoms],
                "self_care": d.self_care,
                "when_to_see_doctor": d.when_to_see_doctor,
                "medicines": [KB.medicines[m]["name"] for m in d.medicines if m in KB.medicines],
            }
            for d in sorted(KB.diseases.values(), key=lambda x: x.name)
        ]
    }


@app.get("/api/medicines")
async def medicines() -> dict[str, Any]:
    return {"medicines": list(KB.medicines.values())}


# ---- condition library ------------------------------------------------------
# Built from open ontologies by scripts/build_condition_library.py. The DB is
# ~80 MB and gitignored, so these endpoints degrade to an empty result on a host
# that has not built it rather than failing. `conditions.available()` says which.
@app.get("/api/conditions/stats")
async def conditions_stats() -> dict[str, Any]:
    """Totals, per-source counts and the licence each source carries."""
    return conditions.stats()


@app.get("/api/conditions/search")
async def conditions_search(
    q: str = "",
    limit: int = 25,
    offset: int = 0,
    source: str | None = None,
    rare: bool = False,
    assessable: bool = False,
) -> dict[str, Any]:
    """Search the library by name, synonym or definition."""
    return conditions.search(
        query=q,
        limit=max(1, min(limit, 100)),
        offset=max(0, offset),
        source=source,
        rare_only=rare,
        detailed_only=assessable,
    )


@app.get("/api/conditions/{condition_id:path}")
async def condition_detail(condition_id: str) -> JSONResponse:
    """One condition with its synonyms and cross-references."""
    record = conditions.get(condition_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no condition with id {condition_id!r}")
    return JSONResponse(record)


@app.get("/api/world-health")
async def world_health(country: str = "IND") -> dict[str, Any]:
    """Real WHO and World Bank indicators, served from the local cache."""
    if not country.isalpha() or len(country) != 3:
        raise HTTPException(status_code=400, detail="country must be a 3-letter ISO code")
    return worlddata.summary(country.upper())


# ---- medicine reference -----------------------------------------------------
# Built from RxNorm by scripts/build_medicine_reference.py. Names and molecule
# links only: no dosing or indication data, so this can confirm a medicine is
# real and what it contains, but never changes which medicine is suggested.
@app.get("/api/drugs/stats")
async def drugs_stats() -> dict[str, Any]:
    return drugref.stats()


@app.get("/api/drugs/search")
async def drugs_search(q: str = "", limit: int = 20, kind: str | None = None) -> dict[str, Any]:
    """Search RxNorm names. `kind` narrows to ingredient, brand or product."""
    if kind and kind not in {"ingredient", "brand", "product", "synonym"}:
        raise HTTPException(status_code=400, detail="unknown kind")
    return drugref.search(q, limit=max(1, min(limit, 100)), kind=kind)


@app.get("/api/drugs/ingredients")
async def drugs_ingredients(name: str) -> dict[str, Any]:
    """Which molecules a medicine name contains, per RxNorm's own relations."""
    if not name.strip():
        raise HTTPException(status_code=400, detail="name is required")
    return {"name": name, "ingredients": drugref.ingredients_for(name)}


@app.get("/api/products/search")
async def product_search(
    q: str = "",
    limit: int = 40,
    combinations: bool = True,
) -> dict[str, Any]:
    """Search the full pharmacy product catalogue.

    This answers "does this product exist, what is it called and what does it
    cost" across every product in the catalogue. It is search only: the
    catalogue carries no clinical data, so a hit here never becomes a
    recommendation. Rows that match a curated medicine carry its id, which the
    client can use to fetch the safe, curated guidance for that medicine.
    """
    limit = max(1, min(limit, 200))
    # Scanning 254k rows takes ~0.5 s of CPU, which would stall every other
    # request if it ran on the event loop.
    results = await run_in_threadpool(
        KB.search_products, q, limit=limit, include_combinations=combinations,
    )
    meta = KB.product_meta
    return {
        "query": q,
        "count": len(results),
        "total_products": meta.get("count", 0),
        "total_molecules": meta.get("molecule_count", 0),
        "source": meta.get("source", ""),
        "results": results,
    }


@app.get("/api/products/stats")
async def product_stats() -> dict[str, Any]:
    """Size and provenance of the full product catalogue."""
    meta = KB.product_meta
    return {
        "products": meta.get("count", 0),
        "molecules": meta.get("molecule_count", 0),
        "rows_read": meta.get("rows_read", 0),
        "source": meta.get("source", ""),
        "note": meta.get("note", ""),
        "curated_medicines": len(KB.medicines),
        "generics_with_brands": len(KB.brands),
        "brands_listed": sum(len(v) for v in KB.brands.values()),
    }


def _split(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]


def _public_dataset_status() -> dict[str, Any]:
    """Dataset shape without the internal build paths.

    The workbook and model locations are server filesystem paths. They are of no
    use to a browser and tell an attacker where the artefacts live, so they are
    dropped from the response rather than shipped to the client.
    """
    data = dataset.status()
    data.pop("path", None)
    return data


def _public_image_model() -> dict[str, Any]:
    """Image-model status without the on-disk model path."""
    data = image_model.status()
    data.pop("path", None)
    return data


def _public_stats(ctx: security.AuthContext) -> dict[str, Any]:
    """Case statistics, with server paths withheld from anonymous callers."""
    data = store.stats()
    data.pop("db_path", None)
    return data


def _run(
    text: str,
    image_bytes: bytes | None,
    top_n: int,
    include_rx: bool,
    age: float | None,
    weight_kg: float | None,
    pregnant: bool,
    chronic_conditions: list[str],
    allergies: list[str],
    consent_share: bool = True,
    sex: str | None = None,
    source: str = "web",
    lang: str = "auto",
    current_medicines: list[str] | None = None,
    image_content_type: str | None = None,
    image_modality: str | None = None,
) -> JSONResponse:
    if not text.strip() and image_bytes is None:
        raise HTTPException(status_code=400, detail="Provide a symptom description, an image, or both.")

    result = pipeline.analyse(
        text=text,
        image_bytes=image_bytes,
        image_content_type=image_content_type,
        top_n=top_n,
        include_rx=include_rx,
        age=age,
        weight_kg=weight_kg,
        pregnant=pregnant,
        chronic_conditions=chronic_conditions,
        allergies=allergies,
        consent_share=consent_share,
        source=source,
        sex=sex,
        lang=lang,
        current_medicines=current_medicines,
        image_modality=image_modality,
    )

    return JSONResponse(result)


@app.post("/api/analyze")
async def analyze_text(
    payload: AnalyzeRequest,
    ctx: security.AuthContext = Depends(security.require_analyze),
) -> JSONResponse:
    """Text-only analysis."""
    return await run_in_threadpool(
        _run,
        text=payload.text,
        image_bytes=None,
        top_n=payload.top_n,
        include_rx=payload.include_rx,
        age=payload.age,
        weight_kg=payload.weight_kg,
        pregnant=payload.pregnant,
        chronic_conditions=_split(payload.chronic_conditions),
        allergies=_split(payload.allergies),
        consent_share=payload.consent_share,
        sex=payload.sex,
        source="api" if not ctx.anonymous else "web",
        lang=payload.lang,
        current_medicines=_split(payload.current_medicines),
    )


@app.post("/api/analyze-image")
async def analyze_image_endpoint(
    image: UploadFile = File(...),
    text: str = Form(""),
    top_n: int = Form(6),
    include_rx: bool = Form(True),
    age: float | None = Form(None),
    weight_kg: float | None = Form(None),
    pregnant: bool = Form(False),
    chronic_conditions: str = Form(""),
    allergies: str = Form(""),
    sex: str | None = Form(None),
    consent_share: bool = Form(True),
    lang: str = Form("auto"),
    current_medicines: str = Form(""),
    image_modality: str | None = Form(None),
    ctx: security.AuthContext = Depends(security.require_analyze),
) -> JSONResponse:
    """Multipart analysis of an uploaded image, optionally with accompanying text."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 12 MB).")
    if not vision.is_supported(data):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported image format: {vision.detect_format(data) or 'unrecognised'}. "
                "Upload any common image (JPEG, PNG, HEIC, WebP, TIFF, BMP, GIF) or a PDF."
            ),
        )

    return await run_in_threadpool(
        _run,
        text=text,
        image_bytes=data,
        image_content_type=image.content_type,
        top_n=top_n,
        include_rx=include_rx,
        age=age,
        weight_kg=weight_kg,
        pregnant=pregnant,
        chronic_conditions=_split(chronic_conditions),
        allergies=_split(allergies),
        consent_share=consent_share,
        sex=sex,
        source="api_image" if not ctx.anonymous else "web_image",
        lang=lang,
        current_medicines=_split(current_medicines),
        image_modality=image_modality,
    )


# ---- learning endpoints -----------------------------------------------------
@app.post("/api/feedback")
async def feedback(
    payload: FeedbackRequest,
    ctx: security.AuthContext = Depends(security.require_feedback),
) -> JSONResponse:
    """Record a correction or rating so the next training run learns from it."""
    try:
        out = store.save_feedback(
            case_id=payload.case_id,
            correct_diagnosis=payload.correct_diagnosis,
            is_correct=payload.is_correct,
            rating=payload.rating,
            comment=payload.comment,
            source="api" if not ctx.anonymous else "web",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(out)


@app.get("/api/model/status")
async def model_status(
    ctx: security.AuthContext = Depends(security.require_read),
) -> dict[str, Any]:
    return {
        "model": learning.status(),
        "image_model": _public_image_model(),
        "data": _public_stats(ctx),
    }


@app.get("/api/data/stats")
async def data_stats(
    ctx: security.AuthContext = Depends(security.require_read),
) -> dict[str, Any]:
    return _public_stats(ctx)


@app.get("/api/export/training-data")
async def export_training_data(
    min_confidence: float = 0.0,
    ctx: security.AuthContext = Depends(security.require_export),
) -> JSONResponse:
    """Export consented, PII-scrubbed labelled examples for offline training in Colab.

    Includes examples derived from consented image uploads (the text read from a
    photographed report), so a Colab run sees the same data the server trains on.
    """
    examples = store.training_examples(min_confidence=min_confidence)
    uploads = store.upload_examples(min_confidence=min_confidence)
    return JSONResponse({
        "count": len(examples) + len(uploads),
        "examples": examples + uploads,
        "uploads": len(uploads),
        "note": "Free text was scrubbed of direct identifiers at capture time. "
                "Review before publishing any dataset.",
    })


@app.post("/api/model/train")
async def train_model(
    payload: TrainRequest,
    background: BackgroundTasks,
    ctx: security.AuthContext = Depends(security.require_train),
) -> JSONResponse:
    """Retrain Dr. Aishu. Runs in the background; poll /api/model/status."""
    from scripts.train_model import collect  # local import keeps startup light

    examples, summary = collect(payload.min_confidence, use_seed=payload.use_seed)
    if not examples:
        raise HTTPException(status_code=400, detail="No training data available.")

    def _job() -> None:
        try:
            if payload.backend == "transformer":
                learning.train_transformer(examples, epochs=payload.epochs)
            else:
                learning.train_tfidf(examples)
        except Exception:
            pass

    background.add_task(_job)
    return JSONResponse({"started": True, "backend": payload.backend, "data": summary})


@app.delete("/api/data/case/{case_id}")
async def delete_case(
    case_id: str,
    ctx: security.AuthContext = Depends(security.require_read),
) -> JSONResponse:
    """Right-to-erasure: purge a case and its feedback."""
    if not store.delete_case(case_id):
        raise HTTPException(status_code=404, detail="case not found")
    return JSONResponse({"deleted": case_id})


# ---- search analytics -------------------------------------------------------
@app.get("/api/analytics/summary")
async def analytics_summary(
    days: int = 30,
    top_n: int = 10,
    include_demo: bool = True,
    ctx: security.AuthContext = Depends(security.require_analyze),
) -> JSONResponse:
    """Aggregated view of recorded searches.

    Counts and averages over structured fields only; free text is never read.
    """
    days = max(1, min(days, 365))
    top_n = max(1, min(top_n, 50))
    return JSONResponse(analytics.summary(days=days, include_demo=include_demo, top_n=top_n))


@app.get("/api/analytics/top-symptoms")
async def analytics_top_symptoms(
    days: int = 30,
    limit: int = 15,
    include_demo: bool = True,
    ctx: security.AuthContext = Depends(security.require_analyze),
) -> JSONResponse:
    data = analytics.summary(days=max(1, min(days, 365)), include_demo=include_demo,
                             top_n=max(1, min(limit, 50)))
    return JSONResponse({"window_days": data["window_days"], "symptoms": data["top_symptoms"]})


@app.get("/api/analytics/top-diagnoses")
async def analytics_top_diagnoses(
    days: int = 30,
    limit: int = 15,
    include_demo: bool = True,
    ctx: security.AuthContext = Depends(security.require_analyze),
) -> JSONResponse:
    data = analytics.summary(days=max(1, min(days, 365)), include_demo=include_demo,
                             top_n=max(1, min(limit, 50)))
    return JSONResponse({"window_days": data["window_days"], "diagnoses": data["top_diagnoses"]})


@app.get("/api/analytics/triage-mix")
async def analytics_triage_mix(
    days: int = 30,
    include_demo: bool = True,
    ctx: security.AuthContext = Depends(security.require_analyze),
) -> JSONResponse:
    data = analytics.summary(days=max(1, min(days, 365)), include_demo=include_demo)
    return JSONResponse({"window_days": data["window_days"], "triage": data["triage_mix"]})


@app.get("/api/analytics/severity")
async def analytics_severity(
    days: int = 30,
    include_demo: bool = True,
    ctx: security.AuthContext = Depends(security.require_analyze),
) -> JSONResponse:
    data = analytics.summary(days=max(1, min(days, 365)), include_demo=include_demo)
    return JSONResponse({"window_days": data["window_days"],
                         "distribution": data["severity_distribution"]})


@app.get("/api/analytics/timeline")
async def analytics_timeline(
    days: int = 30,
    include_demo: bool = True,
    ctx: security.AuthContext = Depends(security.require_analyze),
) -> JSONResponse:
    data = analytics.summary(days=max(1, min(days, 365)), include_demo=include_demo)
    return JSONResponse({"window_days": data["window_days"], "timeline": data["timeline"]})


class DemoRequest(BaseModel):
    count: int = Field(250, ge=1, le=2000)
    days: int = Field(30, ge=1, le=365)
    seed: int | None = None


@app.post("/api/analytics/demo")
async def analytics_demo(
    payload: DemoRequest,
    ctx: security.AuthContext = Depends(security.require_analyze),
) -> JSONResponse:
    """Populate the dashboard with demo searches run through the real pipeline.

    Rows are flagged `is_demo` so they can be told apart from real traffic and
    removed again with DELETE /api/analytics/demo.
    """
    result = demo_data.generate(count=payload.count, days=payload.days, seed=payload.seed)
    return JSONResponse(result)


@app.delete("/api/analytics/demo")
async def analytics_demo_purge(
    ctx: security.AuthContext = Depends(security.require_keys),
) -> JSONResponse:
    """Remove only the generated demo rows."""
    return JSONResponse(demo_data.purge_demo())


@app.get("/api/analytics/brands")
async def analytics_brands(
    limit: int = 20,
    ctx: security.AuthContext = Depends(security.require_read),
) -> JSONResponse:
    """Coverage report for the merged Indian brand catalogue."""
    rows = [
        {
            "medicine_id": mid,
            "medicine": (KB.medicines.get(mid) or {}).get("name", mid),
            "brands": len(entries),
            "cheapest_inr": min(
                (e["price_inr"] for e in entries if e.get("price_inr") is not None),
                default=None,
            ),
        }
        for mid, entries in KB.brands.items()
    ]
    rows.sort(key=lambda r: r["brands"], reverse=True)
    return JSONResponse({
        "source": KB.brand_meta.get("source"),
        "note": KB.brand_meta.get("note"),
        "generics_covered": len(KB.brands),
        "brands_total": sum(len(v) for v in KB.brands.values()),
        "medicines_total": len(KB.medicines),
        "brands": rows[: max(1, min(limit, 200))],
    })


@app.get("/api/brand/{name}")
async def brand_lookup(
    name: str,
    ctx: security.AuthContext = Depends(security.require_read),
) -> JSONResponse:
    """Resolve a brand a user typed (e.g. "Dolo 650") to its generic entry."""
    med_id = KB.resolve_brand(name)
    if not med_id:
        raise HTTPException(status_code=404, detail=f"No brand matching {name!r}.")
    med = KB.medicines.get(med_id) or {}
    return JSONResponse({
        "query": name,
        "medicine_id": med_id,
        "name": med.get("name"),
        "generic": med.get("generic"),
        "drug_class": med.get("drug_class"),
        "otc": med.get("otc"),
        "uses": med.get("uses", []),
        "brands": KB.brands_for(med_id, 8),
    })

