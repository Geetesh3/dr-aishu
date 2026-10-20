"""Adds a `red_flag_level` to each flagged symptom.

`emergency` flags demand immediate emergency care; `urgent` flags require
same-day or next-day clinical review. Grading these separately stops a symptom
such as unintentional weight loss from being treated as a 999-level emergency.
"""
import json
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "app" / "data" / "symptoms.json"

EMERGENCY = {
    "severe_headache", "fainting", "confusion", "neck_stiffness", "slurred_speech",
    "shortness_of_breath", "chest_pain", "severe_abdominal_pain", "suicidal_thoughts",
    "swollen_face_lips", "rapid_breathing", "facial_droop", "dark_skin_patch",
}

data = json.loads(P.read_text())
for s in data["symptoms"]:
    if s.get("red_flag"):
        s["red_flag_level"] = "emergency" if s["id"] in EMERGENCY else "urgent"
    else:
        s["red_flag_level"] = None

P.write_text(json.dumps(data, indent=2) + "\n")

counts: dict[str, int] = {}
for s in data["symptoms"]:
    lvl = s["red_flag_level"] or "none"
    counts[lvl] = counts.get(lvl, 0) + 1
print("red flag levels:", counts)
