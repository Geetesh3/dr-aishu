"""End-to-end orchestration: text and/or image -> full health assessment."""
from __future__ import annotations

from typing import Any

from . import i18n, learning, store
from .analyzer import TRIAGE_META, assess_severity, diagnose
from .kb import KB
from .labs import lab_evidence
from .labs import summarise as lab_evidence_summary
from .nlp import ExtractedFacts, extract
from .prescription import select_medicines
from .routines import build_routine
from .solutions import build_solutions
from .translations import t as translate
from .vision import analyse_image

DISCLAIMER = (
    "This is an educational triage aid, not a medical diagnosis or prescription. "
    "It cannot examine you, run tests or account for your full history. "
    "Always confirm findings with a qualified healthcare professional. "
    "In an emergency, contact your local emergency number immediately."
)


def _localise(result: dict[str, Any], lang: str) -> None:
    """Translate the strings a patient acts on, in place.

    Only the triage block and the disclaimer are translated. Diagnosis names,
    medicine names and routine steps come from the curated English knowledge
    base, and machine-translating a drug name or a clinical finding would create
    a safety risk rather than remove one: a mistranslated drug name is worse than
    an English one the patient can show a pharmacist.
    """
    if lang == "en":
        result["language"] = {"code": "en", "detected": False, "rtl": False}
        return

    meta = i18n.LANGUAGE_BY_CODE.get(lang)
    result["language"] = {
        "code": lang,
        "detected": True,
        "name": meta.english if meta else lang,
        "native": meta.native if meta else lang,
        "rtl": bool(meta and meta.rtl),
    }

    severity = result.get("severity")
    if severity:
        triage = severity.get("triage", "")
        if triage in TRIAGE_META:
            severity["triage_label"] = translate(f"triage.{triage}", lang)
            severity["advice"] = translate(f"triage.{triage}_advice", lang)
        severity["localised"] = True

    result["disclaimer"] = translate("disclaimer", lang)

    medicines = result.get("medicines")
    if medicines and severity and severity.get("triage") == "emergency":
        medicines["general_advice"] = [translate("safety.no_meds_emergency", lang)]


def _merge_facts(base: ExtractedFacts, extra: ExtractedFacts) -> ExtractedFacts:
    """Merge two extraction passes, with `extra` (e.g. OCR text) taking precedence."""
    merged = ExtractedFacts()
    merged.symptoms = dict(base.symptoms)
    merged.symptoms.update(extra.symptoms)
    merged.modifiers = list(dict.fromkeys(base.modifiers + extra.modifiers))
    merged.negated_symptoms = list(dict.fromkeys(base.negated_symptoms + extra.negated_symptoms))
    merged.duration_days = extra.duration_days if extra.duration_days is not None else base.duration_days
    merged.duration_text = extra.duration_text or base.duration_text
    merged.severity_self_reported = (extra.severity_self_reported
                                     if extra.severity_self_reported is not None
                                     else base.severity_self_reported)
    merged.severity_text = extra.severity_text or base.severity_text
    merged.age = extra.age if extra.age is not None else base.age
    merged.weight_kg = extra.weight_kg if extra.weight_kg is not None else base.weight_kg
    merged.temperature_c = extra.temperature_c if extra.temperature_c is not None else base.temperature_c
    merged.oxygen_saturation = (extra.oxygen_saturation if extra.oxygen_saturation is not None
                                else base.oxygen_saturation)
    merged.systolic_bp = extra.systolic_bp if extra.systolic_bp is not None else base.systolic_bp
    merged.diastolic_bp = extra.diastolic_bp if extra.diastolic_bp is not None else base.diastolic_bp
    merged.pulse = extra.pulse if extra.pulse is not None else base.pulse
    merged.is_pregnant = base.is_pregnant or extra.is_pregnant
    merged.chronic_conditions = list(dict.fromkeys(base.chronic_conditions + extra.chronic_conditions))
    merged.allergies = list(dict.fromkeys(base.allergies + extra.allergies))
    merged.current_medicines = list(dict.fromkeys(base.current_medicines + extra.current_medicines))
    # OCR text takes precedence per test, but a value read from the typed note is
    # kept when the report does not repeat it.
    merged.lab_values = {**base.lab_values, **extra.lab_values}
    return merged


def _facts_to_dict(facts: ExtractedFacts) -> dict[str, Any]:
    return {
        "symptoms": [
            {
                "id": sid,
                "label": KB.symptom_label(sid),
                "system": KB.symptoms[sid].system if sid in KB.symptoms else "unknown",
                "red_flag": KB.is_red_flag(sid),
                "certainty": info.get("certainty", "reported"),
                "mentions": info.get("mentions", []),
            }
            for sid, info in facts.symptoms.items()
        ],
        "negated_symptoms": [{"id": s, "label": KB.symptom_label(s)} for s in facts.negated_symptoms],
        "modifiers": [
            {"id": m, "label": next((x["label"] for x in KB.modifiers if x["id"] == m), m)}
            for m in facts.modifiers
        ],
        "duration": {
            "days": facts.duration_days,
            "text": facts.duration_text,
        },
        "self_reported_severity": {
            "score": facts.severity_self_reported,
            "text": facts.severity_text,
        },
        "demographics": {
            "age": facts.age,
            "weight_kg": facts.weight_kg,
            "pregnant": facts.is_pregnant,
        },
        "vitals": {
            "temperature_c": facts.temperature_c,
            "oxygen_saturation": facts.oxygen_saturation,
            "systolic_bp": facts.systolic_bp,
            "diastolic_bp": facts.diastolic_bp,
            "pulse": facts.pulse,
        },
        "history": {
            "chronic_conditions": facts.chronic_conditions,
            "allergies": facts.allergies,
            "current_medicines": facts.current_medicines,
        },
        "labs": lab_evidence_summary(facts.lab_values),
    }


def analyse(
    text: str = "",
    image_bytes: bytes | None = None,
    top_n: int = 6,
    include_rx: bool = True,
    age: float | None = None,
    weight_kg: float | None = None,
    pregnant: bool | None = None,
    chronic_conditions: list[str] | None = None,
    allergies: list[str] | None = None,
    record: bool = True,
    consent_share: bool = False,
    source: str = "web",
    sex: str | None = None,
    lang: str = "auto",
    current_medicines: list[str] | None = None,
    image_content_type: str | None = None,
    image_modality: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline, then persist the result.

    Recording happens here rather than inside the assessment so that every
    outcome is captured, including the no-match and emergency paths, which
    return early. Callers that write the row themselves -- the demo generator,
    which back-dates it -- pass ``record=False``.

    A consented image upload is stored alongside the case so the bytes and the
    text read from them can feed training. The upload is written only after the
    case row exists, so it can be linked to a label.
    """
    result = assess(
        text=text, image_bytes=image_bytes, top_n=top_n, include_rx=include_rx,
        age=age, weight_kg=weight_kg, pregnant=pregnant,
        chronic_conditions=chronic_conditions, allergies=allergies, lang=lang,
        current_medicines=current_medicines, image_modality=image_modality,
    )
    if record:
        stored = store.record_analysis(
            text, result, consent_share=consent_share, source=source, sex=sex,
        )
        result["case"] = stored["case"]
        if image_bytes is not None and (consent_share or store.retains_all()):
            ocr_text = ((result.get("image_analysis") or {}).get("ocr") or {}).get("text", "")
            upload = store.save_upload(
                image_bytes,
                case_id=stored.get("case_id"),
                content_type=image_content_type,
                ocr_text=ocr_text,
                source=source,
                consent_share=consent_share,
                retain=True,
                model_version=result.get("model_version"),
            )
            result["upload"] = upload
    return result


def assess(
    text: str = "",
    image_bytes: bytes | None = None,
    top_n: int = 6,
    include_rx: bool = True,
    age: float | None = None,
    weight_kg: float | None = None,
    pregnant: bool | None = None,
    chronic_conditions: list[str] | None = None,
    allergies: list[str] | None = None,
    lang: str = "auto",
    current_medicines: list[str] | None = None,
    image_modality: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline, then localise the response language.

    ``lang`` is "auto" to detect the language of what the user wrote, or an
    explicit code to force the response language. Detection reads the user's own
    text, never the English preamble the form adds.
    """
    result = _assess_core(
        text=text, image_bytes=image_bytes, top_n=top_n, include_rx=include_rx,
        age=age, weight_kg=weight_kg, pregnant=pregnant,
        chronic_conditions=chronic_conditions, allergies=allergies,
        current_medicines=current_medicines, image_modality=image_modality,
    )

    if lang and lang != "auto":
        chosen = lang if i18n.is_supported(lang) else "en"
        detection = {"code": chosen, "confidence": 1.0, "method": "explicit", "scripts": {}}
    else:
        detection = i18n.detect_language(text)
        chosen = detection["code"]

    _localise(result, chosen)
    result["language"].update({
        "confidence": detection.get("confidence", 0.0),
        "method": detection.get("method", ""),
    })
    return result


def _assess_core(
    text: str = "",
    image_bytes: bytes | None = None,
    top_n: int = 6,
    include_rx: bool = True,
    age: float | None = None,
    weight_kg: float | None = None,
    pregnant: bool | None = None,
    chronic_conditions: list[str] | None = None,
    allergies: list[str] | None = None,
    current_medicines: list[str] | None = None,
    image_modality: str | None = None,
) -> dict[str, Any]:
    """The assessment itself, in English. Pure: reads the KB, writes nothing."""
    result: dict[str, Any] = {
        "disclaimer": DISCLAIMER,
        "inputs": {"text_provided": bool(text.strip()), "image_provided": image_bytes is not None},
    }

    # --- structured form values are prepended so extraction picks them up ---
    preamble_parts: list[str] = []
    if age is not None:
        preamble_parts.append(f"age {age} years")
    if weight_kg is not None:
        preamble_parts.append(f"{weight_kg} kg")
    if pregnant:
        preamble_parts.append("pregnant")
    for c in chronic_conditions or []:
        preamble_parts.append(c)
    for a in allergies or []:
        preamble_parts.append(f"allergic to {a}")
    # Reported medicines are named in the preamble so the prescription layer's
    # interaction check sees them. This is the field that stops a warfarin
    # patient being handed an NSAID.
    for m in current_medicines or []:
        preamble_parts.append(f"taking {m}")
    preamble = ", ".join(preamble_parts)

    combined_text = (preamble + ". " + text).strip() if preamble else text
    facts = extract(combined_text)

    image_result: dict[str, Any] | None = None
    if image_bytes is not None:
        image_result = analyse_image(image_bytes, modality=image_modality)
        result["image_analysis"] = image_result
        if image_result.get("ok"):
            ocr_text = (image_result.get("ocr") or {}).get("text", "")
            if ocr_text:
                ocr_facts = extract(ocr_text)
                facts = _merge_facts(facts, ocr_facts)
            # Fold visual observations in as low-weight symptom evidence.
            for synth in image_result.get("synthetic_symptoms", []):
                sid = synth["symptom_id"]
                if sid in KB.symptoms and sid not in facts.symptoms:
                    facts.symptoms[sid] = {
                        "id": sid,
                        "mentions": [synth["observation"]],
                        "certainty": "visual (image-derived)",
                    }
        else:
            result["image_error"] = image_result.get("error")

    result["extracted"] = _facts_to_dict(facts)

    # Ask Dr. Aishu's trained model for its opinion, if one is loaded.
    symptom_ids = list(facts.symptoms.keys())
    ml_predictions = learning.predict(combined_text, symptom_ids, top_k=top_n) if learning.is_ready() else []
    if ml_predictions:
        result["model_predictions"] = ml_predictions
        result["model_version"] = (learning.status().get("meta") or {}).get("trained_at")

    candidates = diagnose(
        facts, top_n=top_n, ml_predictions=ml_predictions,
        lab_boosts=lab_evidence(facts.lab_values)[0],
    )
    severity = assess_severity(facts, candidates)

    result["diagnoses"] = [
        {
            "id": c.id,
            "name": c.name,
            "category": c.category,
            "icd10": c.icd10,
            "confidence": c.confidence,
            "matched_symptoms": c.matched_symptoms,
            "missing_key_symptoms": c.missing_key_symptoms,
            "description": c.description,
            "contagious": c.contagious,
            "typical_duration_days": c.typical_duration_days,
            "severity_profile": c.severity_profile,
            "self_care": c.self_care,
            "when_to_see_doctor": c.when_to_see_doctor,
            "declared_condition": c.declared_condition,
        }
        for c in candidates
    ]
    result["severity"] = {
        "score": severity.score,
        "label": severity.label,
        "triage": severity.triage,
        "triage_label": severity.triage_label,
        "advice": severity.advice,
        "colour": severity.colour,
        "red_flags": severity.red_flags,
        "factors": severity.factors,
        "duration_assessment": severity.duration_assessment,
    }

    if not candidates:
        result["medicines"] = {
            "recommended": [], "prescription_only": [], "excluded": [],
            "interactions": [], "timetable": [], "general_advice": [
                "No condition could be matched with confidence. Please describe your symptoms in more "
                "detail or consult a clinician."
            ],
        }
        result["routine"] = {
            "name": "General Daily Wellness",
            "focus": KB.routines["general_wellness"]["focus"],
            "blocks": KB.routines["general_wellness"]["blocks"],
            "habits": ["Describe your symptoms to receive a tailored routine."],
            "avoid": [], "hydration_target_l": 2.0, "sleep_target_hours": 8.0, "notes": [],
        }
        result["solutions"] = {
            "problems": [], "condition": None,
            "notes": [
                "No condition could be matched, so there is no per-problem plan. "
                "Describe your symptoms in more detail, or consult a clinician."
            ],
        }
        return result

    if severity.triage == "emergency":
        # Do not suggest medicines in an emergency; the only correct action is
        # to seek immediate emergency care.
        result["medicines"] = {
            "recommended": [], "prescription_only": [],
            "excluded": [],
            "interactions": [],
            "timetable": [],
            "general_advice": [
                "Do not take any new medicine now. Call emergency services or go to the nearest "
                "emergency department immediately.",
                "Note the exact time symptoms started - this guides emergency treatment.",
                "Do not drive yourself. Do not eat or drink in case treatment is needed.",
            ],
        }
        routine = build_routine(candidates, severity, facts, prescription=None)
        result["routine"] = routine.__dict__
        # A per-problem plan still helps in an emergency, but it must not imply
        # self-treatment: no medicine is attached, so nothing appears "covered".
        result["solutions"] = build_solutions(facts, candidates, severity, prescription=None)
        return result

    prescription = select_medicines(candidates, facts, severity.triage, include_rx=include_rx)
    result["medicines"] = {
        "recommended": [m.__dict__ for m in prescription.recommended],
        "prescription_only": [m.__dict__ for m in prescription.prescription_only],
        "excluded": [e.__dict__ for e in prescription.excluded],
        "interactions": prescription.interactions,
        "timetable": prescription.timetable,
        "general_advice": prescription.general_advice,
        "current_medications": [c.__dict__ for c in prescription.current_medications],
        "medication_alerts": [a.__dict__ for a in prescription.medication_alerts],
    }

    routine = build_routine(candidates, severity, facts, prescription)
    result["routine"] = routine.__dict__

    # Per-problem action plan for the leading condition: what to do about each
    # reported symptom, and which of them the medicine plan already covers.
    result["solutions"] = build_solutions(facts, candidates, severity, prescription)

    return result
