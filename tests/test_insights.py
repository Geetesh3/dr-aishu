"""Tests for the Indian medicine catalogue, search analytics and demo data.

The catalogue is a *product* list with no indication column, so the merge has to
add brand names, pack sizes and prices without letting any of that change which
medicine is recommended. Several tests here pin that invariant down, because it
is the difference between a useful brand layer and a system that suggests an
arbitrary drug because its pack was cheap.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app import analytics, demo_data, pipeline, store
from app.kb import KB
from scripts import import_indian_medicines as importer


# ---- importer ----------------------------------------------------------------
def _write_csv(path: Path, rows: list[dict]) -> Path:
    fields = [
        "id", "name", "price(\u20b9)", "Is_discontinued", "manufacturer_name",
        "type", "pack_size_label", "short_composition1", "short_composition2",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})
    return path


def test_base_composition_strips_strength_and_form():
    assert importer._base_composition("Paracetamol (500mg)") == "paracetamol"
    assert importer._base_composition("Amoxycillin  (500mg) ") == "amoxycillin"
    assert importer._base_composition("") == ""


def test_base_composition_strips_salt_suffix():
    # Without this the catalogue's "Chlorpheniramine Maleate" never matches the
    # curated "chlorpheniramine" entry.
    assert importer._base_composition("Chlorpheniramine Maleate") == "chlorpheniramine"
    assert importer._base_composition("Escitalopram Oxalate") == "escitalopram"
    # A lone salt word is left alone rather than reduced to nothing.
    assert importer._base_composition("Sodium") == "sodium"


@pytest.mark.parametrize(
    "comp1,comp2,expected",
    [
        ("Paracetamol (500mg)", "", False),
        ("Paracetamol (500mg)", "Phenylpropanolamine (NA)", True),
        ("Paracetamol (500mg)", "NA", False),
        ("Ambroxol, Levosalbutamol", "", True),
        ("Amoxycillin + Clavulanic Acid", "", True),
    ],
)
def test_combination_detection(comp1, comp2, expected):
    assert importer._is_combination(comp1, comp2) is expected


def test_route_ranking_never_prefers_parenteral_for_oral():
    # A paracetamol vial must never outrank an oral tablet.
    assert importer._form_rank("oral", "oral") < importer._form_rank("injection", "oral")
    # But insulin, which only comes as an injection, must keep its vial.
    assert importer._form_rank("injection", "injection") == 0
    # And a topical generic must keep its cream.
    assert importer._form_rank("topical", "topical") == 0


def test_importer_never_emits_combination_products(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "mini.csv", [
        {"id": "1", "name": "Plainmol 500mg Tablet", "price(\u20b9)": "10",
         "manufacturer_name": "Cipla Ltd", "pack_size_label": "strip of 10 tablets",
         "short_composition1": "Paracetamol (500mg)", "short_composition2": ""},
        {"id": "2", "name": "Coldmol Tablet", "price(\u20b9)": "12",
         "manufacturer_name": "Cipla Ltd", "pack_size_label": "strip of 10 tablets",
         "short_composition1": "Paracetamol (500mg)",
         "short_composition2": "Phenylpropanolamine (NA)"},
    ])
    payload, stats, _unmatched = importer.build_index(csv_path)
    brands = [b["brand"] for b in payload["brands"].get("paracetamol", [])]
    assert "Plainmol 500mg Tablet" in brands
    assert "Coldmol Tablet" not in brands
    assert stats["combination"] == 1


def test_importer_is_deterministic(tmp_path):
    rows = [
        {"id": str(i), "name": f"Brand{i} 500mg Tablet", "price(\u20b9)": str(5 + i),
         "manufacturer_name": "Cipla Ltd", "pack_size_label": "strip of 10 tablets",
         "short_composition1": "Paracetamol (500mg)", "short_composition2": ""}
        for i in range(6)
    ]
    csv_path = _write_csv(tmp_path / "det.csv", rows)
    first, _, _ = importer.build_index(csv_path)
    second, _, _ = importer.build_index(csv_path)
    assert first["brands"] == second["brands"]


# ---- brand layer in the knowledge base ---------------------------------------
def test_brand_index_resolves_known_brands():
    assert KB.resolve_brand("Dolo 650") == "paracetamol"
    assert KB.resolve_brand("Crocin") == "paracetamol"
    assert KB.resolve_brand("Azee 500") == "azithromycin"


def test_brand_resolution_is_case_insensitive():
    assert KB.resolve_brand("dolo 650") == "paracetamol"
    assert KB.resolve_brand("DOLO 650") == "paracetamol"


def test_unknown_brand_resolves_to_none():
    assert KB.resolve_brand("definitely not a real brand 12345") is None
    assert KB.resolve_brand("") is None


def test_brands_are_present_and_shaped():
    brands = KB.brands_for("paracetamol", 5)
    assert brands, "expected Indian paracetamol brands in the merged catalogue"
    for b in brands:
        assert b["brand"]
        assert "price_inr" in b
        assert b["form"] in {"oral", "injection", "inhaler", "eye", "ear", "nasal",
                             "topical", "sublingual", "unknown"}


def test_brand_catalogue_has_no_clinical_fields():
    """The catalogue must not smuggle in indications we did not curate.

    If an indication ever appeared here it could be read as a second opinion on
    what a drug treats, which is exactly the invented-data failure this merge
    was designed to avoid.
    """
    payload = json.loads((Path(KB.__class__.__module__.replace(".", "/"))
                          .parent.parent / "app" / "data" / "indian_brands.json").read_text())
    for entries in payload["brands"].values():
        for entry in entries:
            assert set(entry) <= {"brand", "manufacturer", "pack", "price_inr",
                                  "discontinued", "composition", "form"}, entry


# ---- brand layer cannot change clinical decisions ----------------------------
def test_brands_do_not_change_which_medicine_is_selected():
    """Selection depends on the curated entry only, never on catalogue data."""
    result = pipeline.analyse(text="high fever, body aches and chills for two days")
    plan = result["medicines"]
    for med in plan["recommended"] + plan["prescription_only"]:
        assert med["id"] in KB.medicines
        # Every brand listed must belong to that same curated medicine.
        for brand in med["brands"]:
            assert KB.resolve_brand(brand["brand"]) == med["id"]


def test_emergency_still_recommends_no_medicines_with_brands():
    result = pipeline.analyse(
        text="crushing chest pain spreading to my left arm with sweating",
        age=58,
    )
    assert result["severity"]["triage"] == "emergency"
    assert result["medicines"]["recommended"] == []
    assert result["medicines"]["prescription_only"] == []


# ---- duration regression -----------------------------------------------------
def test_patient_age_is_not_read_as_symptom_duration():
    """The form's age is prepended to the narrative; it must not become a duration.

    Before the fix "age 58 years" matched the bare "<n> years" duration pattern
    and was reported as a 21,170-day illness, which inflated severity and the
    dashboard averages.
    """
    result = pipeline.analyse(text="throbbing headache with nausea", age=58)
    days = (result["extracted"].get("duration") or {}).get("days")
    assert days is None or days <= 365


def test_real_duration_still_parsed_with_age_present():
    result = pipeline.analyse(text="cough and fever for 3 days", age=41)
    assert (result["extracted"].get("duration") or {}).get("days") == 3


# ---- search events -----------------------------------------------------------
def test_every_search_is_recorded_as_an_event():
    pipeline.analyse(text="runny nose and sneezing for three days")
    assert store.stats()["search_events_total"] == 1


def test_event_keeps_no_free_text():
    secret = "my name is Priya Sharma and I have a fever"
    pipeline.analyse(text=secret)
    with store._lock, store._connect() as conn:
        row = dict(conn.execute("SELECT * FROM search_events").fetchone())
    # The narrative must not appear anywhere in the event row.
    assert "Priya" not in json.dumps(row)
    assert row["symptom_count"] >= 1
    assert row["top_diagnosis"]


def test_event_records_structured_signals():
    pipeline.analyse(text="high fever and chills for two days", age=30, sex="male")
    with store._lock, store._connect() as conn:
        row = dict(conn.execute("SELECT * FROM search_events").fetchone())
    assert row["duration_days"] == 2
    assert row["sex"] == "male"
    assert json.loads(row["symptoms"])
    assert row["triage"]


def test_event_training_examples_are_labelled():
    pipeline.analyse(text="burning when I pee and going very often")
    examples = store.event_training_examples()
    assert examples
    for ex in examples:
        assert ex["label"] in KB.diseases
        assert ex["symptoms"]


def test_event_training_examples_respect_min_confidence():
    pipeline.analyse(text="runny nose and sneezing")
    high = store.event_training_examples(min_confidence=100.0)
    assert high == []


def test_training_collect_includes_search_events(monkeypatch):
    from scripts.train_model import collect

    pipeline.analyse(text="sore throat with painful swallowing for two days")
    _examples, summary = collect(0.0, use_seed=False)
    assert summary["search_events"] >= 1


# ---- analytics ---------------------------------------------------------------
def test_analytics_empty_database_is_safe():
    data = analytics.summary(days=30)
    assert data["totals"]["searches"] == 0
    assert data["top_symptoms"] == []
    assert len(data["timeline"]) == 30
    assert all(point["count"] == 0 for point in data["timeline"])


def test_analytics_counts_searches():
    for text in ["runny nose and sneezing", "sore throat and fever", "runny nose and cough"]:
        pipeline.analyse(text=text)
    data = analytics.summary(days=30)
    assert data["totals"]["searches"] == 3
    assert data["totals"]["distinct_symptoms"] >= 1
    assert sum(p["count"] for p in data["timeline"]) == 3


def test_analytics_labels_are_human_readable():
    pipeline.analyse(text="runny nose and sneezing for three days")
    data = analytics.summary(days=30)
    for entry in data["top_symptoms"]:
        assert entry["label"] and entry["label"] != entry["id"]
    for entry in data["top_diagnoses"]:
        assert entry["label"] and " " in entry["label"]


def test_analytics_shares_sum_to_about_100():
    for text in ["runny nose", "sore throat", "fever and chills", "headache"]:
        pipeline.analyse(text=text)
    data = analytics.summary(days=30)
    total_share = sum(d["share"] for d in data["top_diagnoses"])
    assert 95 <= total_share <= 105


def test_analytics_can_exclude_demo_data():
    demo_data.generate(count=5, days=30, seed=1)
    with_demo = analytics.summary(days=30, include_demo=True)["totals"]["searches"]
    without_demo = analytics.summary(days=30, include_demo=False)["totals"]["searches"]
    assert with_demo == 5
    assert without_demo == 0


def test_analytics_window_excludes_old_events():
    demo_data.generate(count=3, days=1, seed=2)
    assert analytics.summary(days=1)["totals"]["searches"] == 3
    assert analytics.summary(days=1, include_demo=True)["totals"]["searches"] >= 3


# ---- demo data ---------------------------------------------------------------
def test_demo_generation_creates_events_and_cases():
    result = demo_data.generate(count=12, days=30, seed=42)
    assert result["created"] == 12
    assert result["failed"] == 0
    assert store.stats()["search_events_demo"] == 12
    assert store.stats()["cases_total"] == 12


def test_demo_generation_is_reproducible():
    first = demo_data.generate(count=10, days=30, seed=99)
    with store._lock, store._connect() as conn:
        rows_a = [dict(r) for r in conn.execute(
            "SELECT symptoms, top_diagnosis FROM search_events ORDER BY id").fetchall()]
    demo_data.purge_demo()
    second = demo_data.generate(count=10, days=30, seed=99)
    with store._lock, store._connect() as conn:
        rows_b = [dict(r) for r in conn.execute(
            "SELECT symptoms, top_diagnosis FROM search_events ORDER BY id").fetchall()]
    assert first["created"] == second["created"] == 10
    assert sorted(map(str, rows_a)) == sorted(map(str, rows_b))


def test_demo_purge_leaves_real_traffic_alone():
    pipeline.analyse(text="runny nose and sneezing for two days")
    demo_data.generate(count=20, days=30, seed=8)
    assert store.stats()["search_events_total"] == 21

    removed = demo_data.purge_demo()
    assert removed["events_removed"] == 20
    assert removed["cases_removed"] == 20
    assert store.stats()["search_events_total"] == 1
    assert store.stats()["cases_total"] == 1


def test_demo_severity_and_triage_are_realistic():
    demo_data.generate(count=120, days=30, seed=17)
    data = analytics.summary(days=30)
    triage = {t["triage"]: t["count"] for t in data["triage_mix"]}
    # Emergencies must be rare but non-zero: the list includes a few red flags.
    assert triage.get("emergency", 0) >= 1
    assert triage.get("emergency", 0) < sum(triage.values()) / 2
    assert triage.get("self_care", 0) > triage.get("emergency", 0)


def test_demo_durations_are_plausible():
    """Guards the age-as-duration bug from reappearing through demo data."""
    demo_data.generate(count=40, days=30, seed=23)
    data = analytics.summary(days=30)
    avg = data["totals"]["avg_duration_days"]
    assert avg is None or avg <= 365


def test_demo_training_examples_are_valid_labels():
    examples = demo_data.demo_training_examples()
    assert examples
    for ex in examples:
        assert ex["label"] in KB.diseases
        assert ex["text"]


def test_demo_does_not_record_consentless_events():
    """Demo rows are consented by construction, so training sees them."""
    demo_data.generate(count=5, days=30, seed=4)
    with store._lock, store._connect() as conn:
        rows = conn.execute("SELECT consent_share FROM search_events").fetchall()
    assert all(r["consent_share"] == 1 for r in rows)


# ---- per-problem action plan -------------------------------------------------
def test_solutions_present_for_matched_condition():
    result = pipeline.assess(text="high fever, body aches and chills with a dry cough for 3 days")
    sol = result["solutions"]
    assert sol["condition"]["name"]
    assert sol["problems"], "expected a per-problem plan"
    for p in sol["problems"]:
        assert p["label"] and p["system"]


def test_solutions_are_grounded_in_the_knowledge_base():
    """Every supportive step must come from the curated map, not be invented."""
    from app.solutions import SYSTEM_SUPPORT

    result = pipeline.assess(text="runny nose, sneezing and a sore throat for three days")
    allowed = {line for lines in SYSTEM_SUPPORT.values() for line in lines}
    for p in result["solutions"]["problems"]:
        for step in p["steps"]:
            assert step in allowed, step


def test_solutions_put_red_flags_first():
    result = pipeline.assess(text="sudden severe headache with vomiting and mild neck ache")
    problems = result["solutions"]["problems"]
    assert problems
    assert problems[0]["is_red_flag"], problems[0]


def test_solutions_never_invent_medicine_for_a_symptom():
    """covered_by may only name medicines that are actually in the plan."""
    result = pipeline.assess(text="fever and cough for two days", age=30)
    plan = result["medicines"]
    names = {m["name"] for m in plan["recommended"] + plan["prescription_only"]}
    for p in result["solutions"]["problems"]:
        for name in p["covered_by"]:
            assert name in names


def test_solutions_attach_no_medicine_in_an_emergency():
    result = pipeline.assess(
        text="crushing chest pain spreading to my left arm with sweating", age=58,
    )
    assert result["severity"]["triage"] == "emergency"
    for p in result["solutions"]["problems"]:
        assert p["covered_by"] == []


# ---- medicine-coverage cache -------------------------------------------------
# The per-request loop used to re-test every symptom against every medicine.
# The result is now cached per (medicine, symptom) pair; these tests hold the
# cache to a direct computation so a stale or partial cache cannot silently
# change a plan.
def test_medicine_coverage_cache_matches_a_direct_computation():
    from app import solutions

    checked = 0
    for med in list(KB.medicines.values())[:40]:
        uses = [KB.normalise(u) for u in (med.get("uses") or [])]
        if not uses:
            continue
        checked += 1
        cached = set(solutions._covered_symptoms(med["name"], uses, KB.symptoms))
        direct = {
            sid for sid in KB.symptoms if solutions._medicine_covers(sid, uses)
        }
        assert cached == direct, med["name"]
    assert checked, "expected at least one medicine with `uses`"


def test_medicine_coverage_cache_is_stable_across_calls():
    from app import solutions

    med = next(
        m for m in KB.medicines.values() if m.get("uses")
    )
    uses = [KB.normalise(u) for u in med["uses"]]
    first = solutions._covered_symptoms(med["name"], uses, KB.symptoms)
    second = solutions._covered_symptoms(med["name"], uses, KB.symptoms)
    assert first == second


def test_medicine_coverage_is_computed_per_reported_symptom():
    """A request must not pay for symptoms it never reported."""
    from app import solutions

    med = next(m for m in KB.medicines.values() if m.get("uses"))
    uses = [KB.normalise(u) for u in med["uses"]]
    key = (med["name"], tuple(uses))
    solutions._COVERS_CACHE.pop(key, None)

    solutions._covered_symptoms(med["name"], uses, ["fever"])
    assert set(solutions._COVERS_CACHE[key]) == {"fever"}


def test_covered_by_still_names_only_planned_medicines():
    """The cached path must keep the grounding guarantee of the uncached one."""
    result = pipeline.assess(text="high fever with body aches and a cough", age=30)
    names = {
        m["name"]
        for m in result["medicines"]["recommended"] + result["medicines"]["prescription_only"]
    }
    for problem in result["solutions"]["problems"]:
        for name in problem["covered_by"]:
            assert name in names


def test_solutions_empty_when_nothing_matches():
    result = pipeline.assess(text="qqq zzz")
    sol = result["solutions"]
    assert sol["problems"] == []
    assert sol["condition"] is None
    assert sol["notes"]


def test_solutions_cap_the_number_of_problems():
    result = pipeline.assess(
        text="fever, cough, headache, nausea, joint pain, rash, sore throat, "
             "fatigue, chills and abdominal pain for two days",
    )
    assert len(result["solutions"]["problems"]) <= 6
