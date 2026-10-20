"""Add natural phrasings for the triage-critical symptoms.

The first pass used one clinical phrasing per symptom. Real descriptions vary,
and for red flags a miss means the emergency is never raised, so the common
everyday variants are added here. Only terms the tests below confirm are absent
are added; nothing existing is removed.
"""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "symptom_terms_i18n.json"

# symptom id -> language -> extra natural phrasings
ADDITIONS: dict[str, dict[str, list[str]]] = {
    "shortness_of_breath": {
        "te": ["శ్వాస తీసుకోవడం కష్టం", "శ్వాస తీసుకోవడం ఇబ్బంది", "ఆయాసం"],
        "hi": ["सांस लेने में तकलीफ", "सांस फूलना", "दम घुटना"],
        "ta": ["மூச்சு விட சிரமம்", "மூச்சு திணறல்"],
        "bn": ["শ্বাস নিতে কষ্ট", "শ্বাসকষ্ট"],
        "mr": ["श्वास घेण्यास त्रास", "दम लागणे"],
        "gu": ["શ્વાસ લેવામાં તકલીફ", "શ્વાસ ચઢવો"],
        "kn": ["ಉಸಿರಾಟದ ತೊಂದರೆ", "ಉಸಿರು ಕಟ್ಟುವುದು"],
        "ml": ["ശ്വാസം എടുക്കാൻ ബുദ്ധിമുട്ട്", "ശ്വാസം മുട്ടൽ"],
        "pa": ["ਸਾਹ ਲੈਣ ਵਿੱਚ ਤਕਲੀਫ", "ਸਾਹ ਫੁੱਲਣਾ"],
        "or": ["ନିଶ୍ୱାସ ନେବାରେ କଷ୍ଟ", "ନିଶ୍ୱାସ ଫୁଲିବା"],
        "ur": ["سانس لینے میں دشواری", "سانس پھولنا"],
        "as": ["উশাহ ল'বলৈ কষ্ট", "নিশাহ ল'বলৈ কষ্ট"],
    },
    "chest_pain": {
        "te": ["ఛాతీలో నొప్పి", "ఛాతీ నొప్పి", "గుండెలో నొప్పి"],
        "hi": ["सीने में दर्द", "छाती में दर्द"],
        "ta": ["மார்பு வலி", "நெஞ்சு வலி"],
        "bn": ["বুকে ব্যথা", "বুক জ্বালা"],
        "mr": ["छातीत दुखणे", "छातीत दुखणे"],
        "gu": ["છાતીમાં દુખાવો"],
        "kn": ["ಎದೆ ನೋವು"],
        "ml": ["നെഞ്ചുവേദന", "നെഞ്ചിൽ വേദന"],
        "pa": ["ਛਾਤੀ ਵਿੱਚ ਦਰਦ"],
        "or": ["ଛାତିରେ ଯନ୍ତ୍ରଣା"],
        "ur": ["سینے میں درد"],
        "as": ["বুকুত বিষ"],
    },
    "high_fever": {
        "te": ["తీవ్ర జ్వరం", "ఎక్కువ జ్వరం"],
        "ta": ["அதிக காய்ச்சல்", "கடும் காய்ச்சல்"],
        "bn": ["প্রচণ্ড জ্বর", "অনেক জ্বর"],
        "mr": ["खूप ताप", "जास्त ताप"],
        "gu": ["તીવ્ર તાવ", "વધારે તાવ"],
        "kn": ["ತೀವ್ರ ಜ್ವರ"],
        "ml": ["കടുത്ത പനി"],
        "pa": ["ਤੇਜ਼ ਬੁਖਾਰ"],
        "or": ["ପ୍ରବଳ ଜ୍ୱର"],
        "ur": ["تیز بخار"],
        "as": ["প্ৰচণ্ড জ্বৰ"],
    },
    "fainting": {
        "te": ["స్పృహ తప్పడం", "మూర్ఛ"],
        "hi": ["बेहोशी", "बेहोश"],
        "ta": ["மயக்கம்", "நினைவிழப்பு"],
        "bn": ["অচেতন", "জ্ঞান হারানো"],
        "mr": ["बेशुद्ध", "बेशुद्धी"],
        "gu": ["બેહોશ"],
        "kn": ["ಪ್ರಜ್ಞೆ ತಪ್ಪುವುದು"],
        "ml": ["ബോധക്ഷയം"],
        "pa": ["ਬੇਹੋਸ਼ੀ"],
        "or": ["ଅଚେତନ"],
        "ur": ["بے ہوشی"],
        "as": ["অচেতন"],
    },
}


def main() -> int:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    added = 0
    for sid, per_lang in ADDITIONS.items():
        for lang, terms in per_lang.items():
            table = data.get(lang)
            if table is None:
                continue
            existing = table.get(sid)
            if existing is None:
                continue
            merged = list(existing) + [t for t in terms if t not in existing]
            if len(merged) != len(existing):
                table[sid] = merged
                added += len(merged) - len(existing)
    PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"added {added} terms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
