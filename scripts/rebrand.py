"""One-off rebrand of the HTML shell to Dr. Aishu."""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "templates" / "index.html"
s = p.read_text()

pairs = [
    ("<title>Health AI — Symptom Analysis, Triage &amp; Care Plan</title>",
     "<title>Dr. Aishu — Medical AI: Symptoms, Triage, Medicines &amp; Daily Routine</title>"),
    ('<span class="brand-text">Health<b>AI</b></span>',
     '<span class="brand-text">Dr.<b>Aishu</b></span>'),
    ("<h1>Understand your symptoms.<br>Get a clear plan for today.</h1>",
     "<h1>Meet Dr. Aishu.<br>Your symptoms, understood.</h1>"),
    ("Health AI identifies likely conditions",
     "Dr. Aishu identifies likely conditions"),
    ("your allergies and conditions, then builds a timed medicine schedule and a daily routine.",
     "your allergies and conditions, then builds a timed medicine schedule and a daily routine.\n      Every case you share helps her learn and improve."),
    ("<p>Health AI — educational triage aid.",
     "<p>Dr. Aishu — educational triage aid."),
    ("<h2>About this tool</h2>", "<h2>About Dr. Aishu</h2>"),
]

for old, new in pairs:
    if old not in s:
        print("MISS:", old[:70])
    s = s.replace(old, new)

p.write_text(s)
print("rebranded; 'Dr. Aishu' occurrences:", s.count("Dr. Aishu") + s.count("Dr.<b>Aishu"))
