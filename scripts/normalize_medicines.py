"""Adds the two remaining DMARDs and normalises all medicine records.

Ensures every record carries the same key set and a consistent key order so
downstream code can rely on the schema.
"""
import json, os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "data", "medicines.json")
d = json.load(open(P))
meds = d["medicines"]

extra = [
    dict(id="methotrexate", name="Methotrexate", generic="methotrexate",
         drug_class="DMARD / antimetabolite", otc=False,
         prescription_note="Disease-modifying drug - specialist supervision and blood monitoring required",
         uses=["rheumatoid arthritis", "psoriatic arthritis", "Crohn's disease"],
         dose_adult="7.5-25 mg", dose_child="specialist dosing",
         max_daily_adult="specialist", max_daily_child="specialist",
         frequency="once weekly (NOT daily)", times=["08:00"], with_food="any", max_days=None,
         age_min=18, pregnancy="contraindicated",
         contraindications=["pregnancy", "breastfeeding", "severe liver disease", "severe kidney disease",
                            "blood dyscrasias", "active infection"],
         side_effects=["nausea", "mouth ulcers", "fatigue", "liver enzyme rise", "reduced blood counts"],
         warnings=["WEEKLY dose only - daily dosing is fatal", "Take folic acid as prescribed",
                   "Regular blood tests required", "Avoid alcohol", "Effective contraception required"],
         interacts_with=["trimethoprim", "ibuprofen", "diclofenac", "amoxicillin", "aspirin"],
         tags=["dmard", "autoimmune"]),
    dict(id="hydroxychloroquine", name="Hydroxychloroquine", generic="hydroxychloroquine",
         drug_class="DMARD / antimalarial", otc=False,
         prescription_note="Disease-modifying drug - specialist supervision required",
         uses=["rheumatoid arthritis", "lupus", "malaria prophylaxis"],
         dose_adult="200-400 mg", dose_child="specialist dosing",
         max_daily_adult="400 mg", max_daily_child="specialist",
         frequency="once daily", times=["08:00"], with_food="after food", max_days=None,
         age_min=18, pregnancy="caution",
         contraindications=["retinal disease", "G6PD deficiency", "psoriasis", "long-standing liver disease"],
         side_effects=["nausea", "rash", "headache", "blurred vision", "skin pigmentation"],
         warnings=["Report any visual disturbance immediately", "Annual eye screening required", "Take with food"],
         interacts_with=["antacid", "ispaghula"], tags=["dmard", "autoimmune"]),
]
meds.extend(extra)

order = ["id", "name", "generic", "drug_class", "otc", "prescription_note", "uses",
         "dose_adult", "dose_child", "max_daily_adult", "max_daily_child", "frequency", "times",
         "with_food", "max_days", "age_min", "pregnancy", "contraindications", "side_effects",
         "warnings", "interacts_with", "tags"]

for m in meds:
    m.setdefault("prescription_note", None)
    for k in order:
        m.setdefault(k, [] if k in ("uses", "times", "contraindications", "side_effects",
                                    "warnings", "interacts_with", "tags") else None)
    for k in ("contraindications", "side_effects", "warnings", "interacts_with", "tags", "uses", "times"):
        if m[k] is None:
            m[k] = []

meds = [{k: m[k] for k in order} for m in meds]
json.dump({"medicines": meds}, open(P, "w"), indent=2)

ids = {m["id"] for m in meds}
dis = json.load(open(os.path.join(os.path.dirname(P), "diseases.json")))["diseases"]
missing = sorted({(x["id"], mid) for x in dis for mid in x["medicines"] if mid not in ids})
print("total:", len(meds), "| uniform keys:", all(set(m.keys()) == set(order) for m in meds))
print("missing refs:", missing if missing else "NONE")
