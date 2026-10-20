"""Adds cardiovascular, neuro/psych, endocrine, supplement and topical medicines.

Imports the base list from build_medicines.py (which defines but does not write),
extends it, and writes app/data/medicines.json.
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_medicines import M, add

# ---------- Cardiovascular ----------
add(id="amlodipine", name="Amlodipine", generic="amlodipine",
    drug_class="Calcium channel blocker", otc=False,
    uses=["hypertension", "angina"],
    dose_adult="5 mg", dose_child="not routine",
    max_daily_adult="10 mg", max_daily_child="n/a",
    frequency="once daily", times=["08:00"], with_food="any",
    max_days=None, age_min=18, pregnancy="caution",
    contraindications=["severe aortic stenosis", "cardiogenic shock", "unstable angina"],
    side_effects=["ankle swelling", "flushing", "headache", "dizziness"],
    warnings=["Take at the same time each day", "Rise slowly to avoid dizziness", "Do not stop suddenly"],
    interacts_with=["clarithromycin", "simvastatin"], tags=["cardiac", "hypertension"])

add(id="losartan", name="Losartan", generic="losartan",
    drug_class="Angiotensin receptor blocker", otc=False,
    uses=["hypertension", "heart failure", "kidney protection in diabetes"],
    dose_adult="50 mg", dose_child="not routine",
    max_daily_adult="100 mg", max_daily_child="n/a",
    frequency="once daily", times=["08:00"], with_food="any",
    max_days=None, age_min=18, pregnancy="contraindicated",
    contraindications=["pregnancy", "bilateral renal artery stenosis", "severe liver disease"],
    side_effects=["dizziness", "fatigue", "raised potassium", "low blood pressure"],
    warnings=["Do NOT use in pregnancy", "Blood tests for kidney function and potassium", "Avoid potassium supplements"],
    interacts_with=["ibuprofen", "diclofenac", "spironolactone", "ramipril"], tags=["cardiac", "hypertension"])

add(id="ramipril", name="Ramipril", generic="ramipril",
    drug_class="ACE inhibitor", otc=False,
    uses=["hypertension", "heart failure", "post-heart-attack protection"],
    dose_adult="2.5-5 mg", dose_child="not routine",
    max_daily_adult="10 mg", max_daily_child="n/a",
    frequency="once daily", times=["08:00"], with_food="any",
    max_days=None, age_min=18, pregnancy="contraindicated",
    contraindications=["pregnancy", "history of angioedema", "bilateral renal artery stenosis"],
    side_effects=["dry persistent cough", "dizziness", "raised potassium", "taste change"],
    warnings=["Do NOT use in pregnancy", "Seek help for facial or tongue swelling", "Monitor kidney function"],
    interacts_with=["ibuprofen", "diclofenac", "spironolactone", "losartan"], tags=["cardiac", "hypertension"])

add(id="hydrochlorothiazide", name="Hydrochlorothiazide", generic="hydrochlorothiazide",
    drug_class="Thiazide diuretic", otc=False,
    uses=["hypertension", "mild fluid overload"],
    dose_adult="12.5-25 mg", dose_child="not routine",
    max_daily_adult="50 mg", max_daily_child="n/a",
    frequency="once daily (morning)", times=["08:00"], with_food="after food",
    max_days=None, age_min=18, pregnancy="avoid",
    contraindications=["anuria", "severe kidney disease", "sulfonamide allergy", "gout"],
    side_effects=["increased urination", "dizziness", "low potassium", "sun sensitivity"],
    warnings=["Take in the morning to avoid night urination", "Monitor potassium and sodium", "Use sun protection"],
    interacts_with=["lithium", "ibuprofen", "digoxin"], tags=["cardiac", "hypertension"])

add(id="furosemide", name="Furosemide (Frusemide)", generic="furosemide",
    drug_class="Loop diuretic", otc=False,
    uses=["heart failure", "fluid overload", "oedema"],
    dose_adult="20-40 mg", dose_child="1-2 mg/kg",
    max_daily_adult="as prescribed", max_daily_child="weight-based",
    frequency="once daily (morning)", times=["08:00"], with_food="any",
    max_days=None, age_min=0, pregnancy="caution",
    contraindications=["anuria", "severe electrolyte depletion", "hepatic coma"],
    side_effects=["frequent urination", "dizziness", "low potassium", "cramps"],
    warnings=["Take in the morning", "Monitor weight and electrolytes daily", "Report rapid weight gain or swelling"],
    interacts_with=["ibuprofen", "diclofenac", "digoxin", "lithium"], tags=["cardiac", "diuretic"])

add(id="spironolactone", name="Spironolactone", generic="spironolactone",
    drug_class="Potassium-sparing diuretic", otc=False,
    uses=["heart failure", "ascites", "resistant hypertension"],
    dose_adult="25 mg", dose_child="1-3 mg/kg",
    max_daily_adult="100 mg", max_daily_child="weight-based",
    frequency="once daily", times=["08:00"], with_food="after food",
    max_days=None, age_min=18, pregnancy="avoid",
    contraindications=["hyperkalaemia", "severe kidney disease", "Addison's disease"],
    side_effects=["raised potassium", "breast tenderness", "dizziness", "irregular periods"],
    warnings=["Monitor potassium closely", "Avoid potassium supplements and salt substitutes"],
    interacts_with=["losartan", "ramipril", "ibuprofen", "diclofenac"], tags=["cardiac", "diuretic"])

add(id="bisoprolol", name="Bisoprolol", generic="bisoprolol",
    drug_class="Beta blocker", otc=False,
    uses=["heart failure", "hypertension", "angina", "rate control"],
    dose_adult="2.5-5 mg", dose_child="not routine",
    max_daily_adult="10 mg", max_daily_child="n/a",
    frequency="once daily", times=["08:00"], with_food="any",
    max_days=None, age_min=18, pregnancy="caution",
    contraindications=["severe asthma", "heart block", "cardiogenic shock", "untreated phaeochromocytoma"],
    side_effects=["fatigue", "cold hands and feet", "slow pulse", "dizziness"],
    warnings=["Do not stop suddenly - rebound angina risk", "May mask hypoglycaemia symptoms in diabetes"],
    interacts_with=["salbutamol", "verapamil", "insulin"], tags=["cardiac"])

add(id="atorvastatin", name="Atorvastatin", generic="atorvastatin",
    drug_class="Statin", otc=False,
    uses=["high cholesterol", "cardiovascular risk reduction"],
    dose_adult="10-80 mg", dose_child="not routine",
    max_daily_adult="80 mg", max_daily_child="n/a",
    frequency="once daily at night", times=["21:00"], with_food="any",
    max_days=None, age_min=18, pregnancy="contraindicated",
    contraindications=["pregnancy", "active liver disease", "breastfeeding"],
    side_effects=["muscle aches", "headache", "raised liver enzymes", "nausea"],
    warnings=["Report unexplained muscle pain or weakness", "Avoid grapefruit juice", "Liver function monitoring"],
    interacts_with=["clarithromycin", "fluconazole", "colchicine"], tags=["cardiac", "cholesterol"])

add(id="nitroglycerin", name="Nitroglycerin (GTN) sublingual", generic="glyceryl trinitrate",
    drug_class="Nitrate vasodilator", otc=False,
    uses=["angina attack", "angina prevention before exertion"],
    dose_adult="0.4-0.8 mg", dose_child="not routine",
    max_daily_adult="as needed", max_daily_child="n/a",
    frequency="as needed", times=["as needed"], with_food="any",
    max_days=None, age_min=18, pregnancy="caution",
    contraindications=["severe hypotension", "concurrent PDE5 inhibitors", "aortic stenosis"],
    side_effects=["headache", "flushing", "dizziness", "low blood pressure"],
    warnings=["Sit down before taking - may cause faintness", "If pain persists after 3 doses over 15 minutes, call emergency services", "Never combine with erectile dysfunction drugs"],
    interacts_with=[], tags=["cardiac", "emergency"])

# ---------- Neuro / Psych ----------
add(id="sumatriptan", name="Sumatriptan", generic="sumatriptan",
    drug_class="Triptan (5-HT1 agonist)", otc=False,
    uses=["acute migraine attack"],
    dose_adult="50-100 mg", dose_child="not routine",
    max_daily_adult="300 mg", max_daily_child="n/a",
    frequency="at attack onset", times=["as needed"], with_food="any",
    max_days=10, age_min=18, pregnancy="caution",
    contraindications=["ischaemic heart disease", "stroke history", "uncontrolled hypertension", "ergotamine use"],
    side_effects=["chest tightness", "tingling", "dizziness", "drowsiness"],
    warnings=["Take as early as possible in the attack", "Not for prevention", "Do not use more than 10 days a month"],
    interacts_with=["sertraline", "fluoxetine", "escitalopram"], tags=["migraine"])

add(id="amitriptyline", name="Amitriptyline", generic="amitriptyline",
    drug_class="Tricyclic antidepressant", otc=False,
    uses=["tension headache prevention", "migraine prevention", "neuropathic pain", "fibromyalgia"],
    dose_adult="10-25 mg", dose_child="not routine",
    max_daily_adult="75 mg", max_daily_child="n/a",
    frequency="once daily at night", times=["21:00"], with_food="any",
    max_days=None, age_min=18, pregnancy="caution",
    contraindications=["recent heart attack", "arrhythmias", "narrow-angle glaucoma", "prostate enlargement"],
    side_effects=["drowsiness", "dry mouth", "constipation", "weight gain", "blurred vision"],
    warnings=["Take at night due to sedation", "Do not stop suddenly", "Avoid alcohol"],
    interacts_with=["cetirizine", "chlorpheniramine", "tramadol", "terbinafine"], tags=["neuro", "prevention"])

add(id="sertraline", name="Sertraline", generic="sertraline",
    drug_class="SSRI antidepressant", otc=False,
    uses=["depression", "anxiety", "panic disorder", "OCD", "PTSD"],
    dose_adult="50 mg", dose_child="not routine",
    max_daily_adult="200 mg", max_daily_child="n/a",
    frequency="once daily", times=["08:00"], with_food="after food",
    max_days=None, age_min=18, pregnancy="caution",
    contraindications=["MAOI use", "concurrent pimozide", "hypersensitivity"],
    side_effects=["nausea", "insomnia", "headache", "sexual dysfunction", "initial anxiety increase"],
    warnings=["Takes 2-4 weeks to work", "Do not stop abruptly", "Seek help if mood worsens or for any thoughts of self-harm"],
    interacts_with=["metoclopramide", "sumatriptan", "ibuprofen", "tramadol", "dextromethorphan"], tags=["psych"])

add(id="fluoxetine", name="Fluoxetine", generic="fluoxetine",
    drug_class="SSRI antidepressant", otc=False,
    uses=["depression", "bulimia", "OCD", "panic disorder"],
    dose_adult="20 mg", dose_child="10 mg",
    max_daily_adult="60 mg", max_daily_child="20 mg",
    frequency="once daily (morning)", times=["08:00"], with_food="any",
    max_days=None, age_min=8, pregnancy="caution",
    contraindications=["MAOI use", "thioridazine use", "hypersensitivity"],
    side_effects=["nausea", "insomnia", "headache", "restlessness", "sexual dysfunction"],
    warnings=["Take in the morning - can be activating", "Do not stop abruptly", "Seek help for any thoughts of self-harm"],
    interacts_with=["metoclopramide", "dextromethorphan", "ibuprofen", "tramadol"], tags=["psych"])

add(id="escitalopram", name="Escitalopram", generic="escitalopram",
    drug_class="SSRI antidepressant", otc=False,
    uses=["depression", "generalised anxiety", "panic disorder"],
    dose_adult="10 mg", dose_child="not routine",
    max_daily_adult="20 mg", max_daily_child="n/a",
    frequency="once daily", times=["08:00"], with_food="any",
    max_days=None, age_min=18, pregnancy="caution",
    contraindications=["MAOI use", "prolonged QT", "hypersensitivity"],
    side_effects=["nausea", "headache", "sleep disturbance", "sexual dysfunction"],
    warnings=["Do not stop abruptly", "Avoid with other QT-prolonging drugs", "Seek help for thoughts of self-harm"],
    interacts_with=["metoclopramide", "sumatriptan", "domperidone", "clarithromycin"], tags=["psych"])

add(id="mirtazapine", name="Mirtazapine", generic="mirtazapine",
    drug_class="Noradrenergic antidepressant", otc=False,
    uses=["depression with insomnia", "depression with poor appetite"],
    dose_adult="15 mg", dose_child="not routine",
    max_daily_adult="45 mg", max_daily_child="n/a",
    frequency="once daily at night", times=["21:00"], with_food="any",
    max_days=None, age_min=18, pregnancy="caution",
    contraindications=["MAOI use", "hypersensitivity"],
    side_effects=["sedation", "increased appetite", "weight gain", "dry mouth"],
    warnings=["Take at night", "Do not stop abruptly", "Seek help for thoughts of self-harm"],
    interacts_with=["tramadol", "amitriptyline", "alcohol"], tags=["psych"])

add(id="duloxetine", name="Duloxetine", generic="duloxetine",
    drug_class="SNRI antidepressant", otc=False,
    uses=["fibromyalgia", "neuropathic pain", "depression", "anxiety"],
    dose_adult="30 mg", dose_child="not routine",
    max_daily_adult="120 mg", max_daily_child="n/a",
    frequency="once daily", times=["08:00"], with_food="after food",
    max_days=None, age_min=18, pregnancy="caution",
    contraindications=["MAOI use", "uncontrolled narrow-angle glaucoma", "severe liver disease"],
    side_effects=["nausea", "dry mouth", "drowsiness", "sweating", "raised blood pressure"],
    warnings=["Do not stop abruptly", "Monitor blood pressure", "Avoid alcohol"],
    interacts_with=["chlorpheniramine", "tramadol", "ibuprofen"], tags=["psych", "pain"])

add(id="pregabalin", name="Pregabalin", generic="pregabalin",
    drug_class="Anticonvulsant / neuropathic analgesic", otc=False,
    uses=["neuropathic pain", "fibromyalgia", "generalised anxiety", "seizures"],
    dose_adult="75 mg", dose_child="not routine",
    max_daily_adult="600 mg", max_daily_child="n/a",
    frequency="twice daily", times=["08:00", "20:00"], with_food="any",
    max_days=None, age_min=18, pregnancy="avoid",
    contraindications=["hypersensitivity", "severe heart failure"],
    side_effects=["drowsiness", "dizziness", "weight gain", "blurred vision", "swelling"],
    warnings=["May impair driving", "Do not stop suddenly", "Avoid alcohol", "Potential for dependence"],
    interacts_with=["cetirizine", "chlorpheniramine", "tramadol", "alcohol"], tags=["neuro", "pain"])

add(id="propranolol", name="Propranolol", generic="propranolol",
    drug_class="Beta blocker", otc=False,
    uses=["anxiety with palpitations", "migraine prevention", "tremor", "hyperthyroidism symptoms"],
    dose_adult="10-40 mg", dose_child="not routine",
    max_daily_adult="160 mg", max_daily_child="n/a",
    frequency="three times daily", times=["08:00", "14:00", "20:00"], with_food="before food",
    max_days=None, age_min=18, pregnancy="caution",
    contraindications=["severe asthma", "heart block", "cardiogenic shock", "untreated heart failure"],
    side_effects=["fatigue", "cold extremities", "slow pulse", "dizziness", "sleep disturbance"],
    warnings=["Do not stop suddenly", "Avoid in asthma", "Can mask hypoglycaemia symptoms"],
    interacts_with=["salbutamol", "insulin", "verapamil"], tags=["psych", "cardiac"])

# ---------- Endocrine / metabolic ----------
add(id="levothyroxine", name="Levothyroxine", generic="levothyroxine",
    drug_class="Thyroid hormone", otc=False,
    uses=["hypothyroidism", "thyroid hormone replacement"],
    dose_adult="25-100 mcg", dose_child="weight-based",
    max_daily_adult="200 mcg", max_daily_child="weight-based",
    frequency="once daily", times=["07:00"], with_food="before food",
    max_days=None, age_min=0, pregnancy="safe",
    contraindications=["untreated hyperthyroidism", "untreated adrenal insufficiency", "recent heart attack"],
    side_effects=["palpitations", "weight loss", "insomnia", "heat intolerance if over-replaced"],
    warnings=["Take on an empty stomach 30-60 minutes before breakfast", "Separate from iron, calcium and antacids by 4 hours", "Regular thyroid blood tests"],
    interacts_with=["ferrous_sulfate", "antacid", "ispaghula", "omeprazole"], tags=["endocrine"])

add(id="carbimazole", name="Carbimazole", generic="carbimazole",
    drug_class="Antithyroid agent", otc=False,
    uses=["hyperthyroidism", "Graves' disease"],
    dose_adult="10-20 mg", dose_child="weight-based",
    max_daily_adult="60 mg", max_daily_child="weight-based",
    frequency="once daily", times=["08:00"], with_food="after food",
    max_days=None, age_min=0, pregnancy="caution",
    contraindications=["severe blood disorders", "previous agranulocytosis with this drug", "severe liver disease"],
    side_effects=["rash", "nausea", "itching", "reduced white cells (rare)"],
    warnings=["Report fever or sore throat immediately - may indicate low white cells", "Regular blood tests required"],
    interacts_with=["warfarin", "ispaghula"], tags=["endocrine"])

add(id="metformin", name="Metformin", generic="metformin",
    drug_class="Biguanide antidiabetic", otc=False,
    uses=["type 2 diabetes", "insulin resistance", "PCOS"],
    dose_adult="500 mg", dose_child="not routine",
    max_daily_adult="2550 mg", max_daily_child="n/a",
    frequency="twice daily", times=["08:00", "20:00"], with_food="after food",
    max_days=None, age_min=10, pregnancy="caution",
    contraindications=["severe kidney disease", "lactic acidosis history", "severe liver disease", "acute heart failure"],
    side_effects=["nausea", "diarrhoea", "abdominal discomfort", "metallic taste"],
    warnings=["Take with or just after food to reduce stomach upset", "Stop before contrast imaging or surgery", "Report muscle pain or breathlessness"],
    interacts_with=["prednisolone", "furosemide", "alcohol"], tags=["diabetes"])

add(id="glimepiride", name="Glimepiride", generic="glimepiride",
    drug_class="Sulfonylurea antidiabetic", otc=False,
    uses=["type 2 diabetes"],
    dose_adult="1-4 mg", dose_child="not routine",
    max_daily_adult="6 mg", max_daily_child="n/a",
    frequency="once daily", times=["07:30"], with_food="before food",
    max_days=None, age_min=18, pregnancy="avoid",
    contraindications=["type 1 diabetes", "severe kidney or liver disease", "diabetic ketoacidosis"],
    side_effects=["hypoglycaemia", "weight gain", "nausea", "rash"],
    warnings=["Take just before breakfast", "Do not skip meals - hypoglycaemia risk", "Carry glucose for low sugar"],
    interacts_with=["propranolol", "bisoprolol", "alcohol"], tags=["diabetes"])

add(id="insulin", name="Insulin", generic="insulin",
    drug_class="Antidiabetic hormone", otc=False,
    prescription_note="Injectable - prescribed and dose-adjusted by a clinician",
    uses=["type 1 diabetes", "type 2 diabetes with poor control", "diabetic emergencies"],
    dose_adult="individualised", dose_child="individualised",
    max_daily_adult="individualised", max_daily_child="individualised",
    frequency="as prescribed", times=["as prescribed"], with_food="relative to meals",
    max_days=None, age_min=0, pregnancy="safe",
    contraindications=["hypoglycaemia", "hypersensitivity to the formulation"],
    side_effects=["hypoglycaemia", "weight gain", "injection site reactions"],
    warnings=["Dose is individual - never adjust without advice", "Rotate injection sites", "Recognise and treat hypoglycaemia"],
    interacts_with=["propranolol", "bisoprolol", "prednisolone"], tags=["diabetes"])

# ---------- Haematology / supplements ----------
add(id="ferrous_sulfate", name="Ferrous Sulfate (iron)", generic="ferrous sulfate",
    drug_class="Iron supplement", otc=True,
    uses=["iron deficiency anaemia", "iron deficiency"],
    dose_adult="200 mg (65 mg elemental)", dose_child="3-6 mg/kg/day",
    max_daily_adult="600 mg", max_daily_child="weight-based",
    frequency="once to twice daily", times=["08:00", "20:00"], with_food="before food",
    max_days=None, age_min=0, pregnancy="safe",
    contraindications=["iron overload", "haemochromatosis", "repeated blood transfusions", "active peptic ulcer"],
    side_effects=["constipation", "black stools", "nausea", "abdominal pain"],
    warnings=["Black stools are expected and harmless", "Take with vitamin C to aid absorption", "Separate from tea, calcium and antacids", "Keep away from children - overdose is dangerous"],
    interacts_with=["antacid", "levothyroxine", "ciprofloxacin", "zinc", "ispaghula"], tags=["supplement", "anaemia"])

add(id="folic_acid", name="Folic Acid", generic="folic acid",
    drug_class="Vitamin supplement", otc=True,
    uses=["folate deficiency anaemia", "pregnancy supplementation"],
    dose_adult="400-5000 mcg", dose_child="weight-based",
    max_daily_adult="5000 mcg", max_daily_child="weight-based",
    frequency="once daily", times=["08:00"], with_food="any",
    max_days=None, age_min=0, pregnancy="safe",
    contraindications=["untreated B12 deficiency (can mask it)", "hypersensitivity"],
    side_effects=["nausea", "bloating", "bitter taste (rare)"],
    warnings=["Rule out B12 deficiency before high-dose treatment", "Essential before and in early pregnancy"],
    interacts_with=["methotrexate", "trimethoprim"], tags=["supplement"])

add(id="vitamin_b12", name="Vitamin B12 (Cyanocobalamin)", generic="cyanocobalamin",
    drug_class="Vitamin supplement", otc=True,
    uses=["B12 deficiency", "pernicious anaemia", "neurological symptoms of deficiency"],
    dose_adult="1000 mcg", dose_child="weight-based",
    max_daily_adult="1000 mcg", max_daily_child="weight-based",
    frequency="once daily", times=["08:00"], with_food="any",
    max_days=None, age_min=0, pregnancy="safe",
    contraindications=["hypersensitivity", "Leber's disease"],
    side_effects=["nausea", "headache", "injection site pain"],
    warnings=["Oral absorption is poor in pernicious anaemia - injections may be needed", "Long-term metformin users may need supplementation"],
    interacts_with=["omeprazole", "metformin"], tags=["supplement"])

add(id="vitamin_c", name="Vitamin C (Ascorbic Acid)", generic="ascorbic acid",
    drug_class="Vitamin supplement", otc=True,
    uses=["immune support", "cold symptom support", "iron absorption"],
    dose_adult="500-1000 mg", dose_child="250 mg",
    max_daily_adult="2000 mg", max_daily_child="500 mg",
    frequency="once daily", times=["08:00"], with_food="after food",
    max_days=30, age_min=1, pregnancy="safe",
    contraindications=["kidney stone history (high dose)", "haemochromatosis", "G6PD deficiency (high dose)"],
    side_effects=["stomach upset", "diarrhoea at high doses"],
    warnings=["High doses may cause kidney stones in susceptible people", "Take with food"],
    interacts_with=["ferrous_sulfate", "warfarin"], tags=["supplement"])

add(id="vitamin_d", name="Vitamin D3 (Cholecalciferol)", generic="cholecalciferol",
    drug_class="Vitamin supplement", otc=True,
    uses=["vitamin D deficiency", "bone health", "immune support"],
    dose_adult="1000-2000 IU", dose_child="400-1000 IU",
    max_daily_adult="4000 IU", max_daily_child="2000 IU",
    frequency="once daily", times=["08:00"], with_food="after food",
    max_days=None, age_min=0, pregnancy="safe",
    contraindications=["hypercalcaemia", "sarcoidosis", "hyperparathyroidism"],
    side_effects=["nausea (overdose)", "raised calcium (overdose)"],
    warnings=["Take with a meal containing fat for absorption", "Do not exceed the recommended dose"],
    interacts_with=["hydrochlorothiazide"], tags=["supplement"])

add(id="glucosamine", name="Glucosamine", generic="glucosamine sulfate",
    drug_class="Supplement", otc=True,
    uses=["osteoarthritis symptom support"],
    dose_adult="500 mg", dose_child="not recommended",
    max_daily_adult="1500 mg", max_daily_child="n/a",
    frequency="three times daily", times=["08:00", "14:00", "20:00"], with_food="after food",
    max_days=90, age_min=18, pregnancy="avoid",
    contraindications=["shellfish allergy", "diabetes (monitor glucose)"],
    side_effects=["stomach upset", "drowsiness", "headache"],
    warnings=["Evidence for benefit is modest", "May raise blood sugar - monitor in diabetes"],
    interacts_with=["warfarin"], tags=["supplement", "joint"])

# ---------- Gout / urology ----------
add(id="colchicine", name="Colchicine", generic="colchicine",
    drug_class="Anti-gout agent", otc=False,
    uses=["acute gout attack", "gout prevention"],
    dose_adult="500 mcg", dose_child="not routine",
    max_daily_adult="2500 mcg", max_daily_child="n/a",
    frequency="two to four times daily", times=["08:00", "14:00", "20:00"], with_food="after food",
    max_days=5, age_min=18, pregnancy="avoid",
    contraindications=["severe kidney disease", "severe liver disease", "blood dyscrasias", "concurrent clarithromycin"],
    side_effects=["diarrhoea", "nausea", "vomiting", "abdominal pain"],
    warnings=["Stop at the first sign of diarrhoea", "Narrow therapeutic index - do not exceed the dose", "Check interactions carefully"],
    interacts_with=["clarithromycin", "atorvastatin", "fluconazole", "ibuprofen"], tags=["gout"])

add(id="allopurinol", name="Allopurinol", generic="allopurinol",
    drug_class="Xanthine oxidase inhibitor", otc=False,
    uses=["gout prevention", "urate lowering"],
    dose_adult="100-300 mg", dose_child="weight-based",
    max_daily_adult="900 mg", max_daily_child="weight-based",
    frequency="once daily", times=["08:00"], with_food="after food",
    max_days=None, age_min=0, pregnancy="caution",
    contraindications=["previous allopurinol hypersensitivity", "acute gout attack (start after settling)"],
    side_effects=["rash", "nausea", "raised liver enzymes", "acute gout flare on starting"],
    warnings=["Do not start during an acute attack", "Report any rash immediately - stop the drug", "Drink plenty of water"],
    interacts_with=["amoxicillin", "azathioprine", "warfarin"], tags=["gout"])

add(id="tamsulosin", name="Tamsulosin", generic="tamsulosin",
    drug_class="Alpha blocker", otc=False,
    uses=["kidney stones (expulsion)", "benign prostatic hyperplasia"],
    dose_adult="0.4 mg", dose_child="not routine",
    max_daily_adult="0.8 mg", max_daily_child="n/a",
    frequency="once daily", times=["20:00"], with_food="after food",
    max_days=28, age_min=18, pregnancy="not applicable",
    contraindications=["postural hypotension", "severe liver disease", "cataract surgery planned"],
    side_effects=["dizziness", "retrograde ejaculation", "headache", "low blood pressure"],
    warnings=["Take after the same meal each day", "May cause dizziness on standing", "Tell your eye surgeon before cataract surgery"],
    interacts_with=["amlodipine", "sildenafil"], tags=["urology"])

add(id="cranberry", name="Cranberry supplement", generic="cranberry extract",
    drug_class="Herbal supplement", otc=True,
    uses=["UTI prevention support"],
    dose_adult="1 capsule or 250 mL juice", dose_child="not recommended",
    max_daily_adult="2 doses", max_daily_child="n/a",
    frequency="twice daily", times=["08:00", "20:00"], with_food="after food",
    max_days=90, age_min=18, pregnancy="safe",
    contraindications=["warfarin use", "kidney stone history (oxalate)"],
    side_effects=["stomach upset", "diarrhoea"],
    warnings=["Prevention only - does not treat an active infection", "Interacts with warfarin"],
    interacts_with=["warfarin"], tags=["supplement", "urinary"])

# ---------- Topical / eye ----------
add(id="hydrocortisone", name="Hydrocortisone 1% cream", generic="hydrocortisone",
    drug_class="Topical corticosteroid (mild)", otc=True,
    uses=["eczema", "mild dermatitis", "insect bite inflammation", "mild allergic skin rash"],
    dose_adult="thin layer", dose_child="thin layer",
    max_daily_adult="2 applications", max_daily_child="2 applications",
    frequency="once to twice daily", times=["08:00", "20:00"], with_food="topical",
    max_days=7, age_min=0, pregnancy="safe",
    contraindications=["untreated skin infection", "acne", "rosacea", "face unless advised"],
    side_effects=["skin thinning with prolonged use", "burning", "itching"],
    warnings=["Apply thinly - do not use on the face for long periods", "Do not use on infected skin", "Maximum 7 days without review"],
    interacts_with=[], tags=["topical", "skin"])

add(id="emollient", name="Emollient (moisturiser)", generic="emollient",
    drug_class="Skin barrier agent", otc=True,
    uses=["eczema", "dry skin", "atopic dermatitis"],
    dose_adult="apply liberally", dose_child="apply liberally",
    max_daily_adult="as needed", max_daily_child="as needed",
    frequency="at least twice daily", times=["08:00", "20:00"], with_food="topical",
    max_days=None, age_min=0, pregnancy="safe",
    contraindications=["hypersensitivity to ingredients"],
    side_effects=["transient stinging", "greasiness"],
    warnings=["Apply within 3 minutes of bathing to lock in moisture", "Use plenty - emollients are the foundation of eczema care"],
    interacts_with=[], tags=["topical", "skin"])

add(id="tacrolimus", name="Tacrolimus ointment", generic="tacrolimus",
    drug_class="Topical calcineurin inhibitor", otc=False,
    uses=["eczema (moderate to severe)", "atopic dermatitis"],
    dose_adult="thin layer 0.03-0.1%", dose_child="thin layer 0.03%",
    max_daily_adult="2 applications", max_daily_child="2 applications",
    frequency="twice daily", times=["08:00", "20:00"], with_food="topical",
    max_days=42, age_min=2, pregnancy="caution",
    contraindications=["infected skin", "Netherton syndrome", "hypersensitivity"],
    side_effects=["burning on application", "itching", "increased sun sensitivity"],
    warnings=["Avoid direct sunlight", "Do not apply to infected skin", "Use only as prescribed"],
    interacts_with=[], tags=["topical", "skin"])

add(id="silver_sulfadiazine", name="Silver Sulfadiazine cream", generic="silver sulfadiazine",
    drug_class="Topical antibacterial (burns)", otc=False,
    uses=["partial thickness burns", "burn wound infection prevention"],
    dose_adult="thin layer", dose_child="thin layer",
    max_daily_adult="1-2 applications", max_daily_child="1-2 applications",
    frequency="once to twice daily", times=["08:00", "20:00"], with_food="topical",
    max_days=14, age_min=2, pregnancy="avoid",
    contraindications=["sulfonamide allergy", "G6PD deficiency", "premature infants", "pregnancy"],
    side_effects=["burning", "itching", "skin discolouration"],
    warnings=["For burn wounds only", "Avoid in sulfonamide allergy", "Large burns need hospital care"],
    interacts_with=[], tags=["topical", "burn"])

add(id="chloramphenicol_eye", name="Chloramphenicol eye drops", generic="chloramphenicol",
    drug_class="Topical antibiotic (eye)", otc=True,
    uses=["bacterial conjunctivitis", "eye infection"],
    dose_adult="1 drop", dose_child="1 drop",
    max_daily_adult="4 drops", max_daily_child="4 drops",
    frequency="four times daily", times=["08:00", "12:00", "16:00", "20:00"], with_food="topical",
    max_days=5, age_min=0, pregnancy="safe",
    contraindications=["chloramphenicol allergy", "previous blood disorder with chloramphenicol", "contact lens wear"],
    side_effects=["stinging", "brief blurred vision", "eye irritation"],
    warnings=["Do not wear contact lenses during treatment", "Stop if no improvement in 5 days", "Do not share eye drops"],
    interacts_with=[], tags=["topical", "eye"])

add(id="artificial_tears", name="Artificial tears (hypromellose)", generic="hypromellose",
    drug_class="Ocular lubricant", otc=True,
    uses=["dry eye", "eye irritation", "conjunctivitis comfort"],
    dose_adult="1 drop", dose_child="1 drop",
    max_daily_adult="as needed", max_daily_child="as needed",
    frequency="as needed", times=["as needed"], with_food="topical",
    max_days=None, age_min=0, pregnancy="safe",
    contraindications=["hypersensitivity to preservatives"],
    side_effects=["temporary blurring", "mild stinging"],
    warnings=["Use preservative-free versions if used more than 4 times daily", "Do not touch the dropper tip to the eye"],
    interacts_with=[], tags=["topical", "eye"])

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app", "data", "medicines.json")
with open(out, "w") as f:
    json.dump({"medicines": M}, f, indent=2)
print("wrote", len(M), "medicines to", os.path.normpath(out))
