"""End-to-end test of the Dr. Aishu learning loop against a live server.

Exercises: analysis with consent, PII scrubbing, feedback correction, training
data export, and the retrain endpoint.

The export, feedback and retrain endpoints require an API key, so pass one in
`DR_AISHU_API_KEY` (create it with `python -m app.keys create` or the signup
endpoint). Without a key the script stops at step 2 with a 403.

Usage:  DR_AISHU_API_KEY=... python scripts/test_learning_loop.py [base_url]
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:12000"
API_KEY = os.environ.get("DR_AISHU_API_KEY", "")


def _headers(content_type: str | None = None) -> dict[str, str]:
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers=_headers("application/json"),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def get(path: str) -> dict:
    req = urllib.request.Request(BASE + path, headers=_headers(), method="GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def delete(path: str) -> dict:
    req = urllib.request.Request(BASE + path, headers=_headers(), method="DELETE")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> int:
    ok = True

    if not API_KEY:
        print(
            "No API key set. The export, feedback and retrain endpoints need one.\n"
            "Create a key, then re-run:\n"
            "  DR_AISHU_API_KEY=<key> python scripts/test_learning_loop.py [base_url]",
            file=sys.stderr,
        )
        return 2

    try:
        return _run(ok)
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            print(f"\nAuth failed ({exc.code}) on {exc.url}. Check DR_AISHU_API_KEY "
                  "and that the key has read/export/train scopes.", file=sys.stderr)
            return 2
        raise


def _run(ok: bool) -> int:

    print("1. Analyse with consent, including deliberate PII...")
    dirty = (
        "My name is Priya Sharma and my phone is 9876543210, email priya.sharma@example.com. "
        "I have had a high fever and severe body ache and chills for 3 days with a dry cough. "
        "Temperature 39.5C. Age 30."
    )
    res = post("/api/analyze", {"text": dirty, "consent_share": True, "age": 30})
    case_id = res["case"]["case_id"]
    top = res["diagnoses"][0]
    print(f"   case_id={case_id} stored={res['case']['stored_for_training']} "
          f"redactions={res['case']['redactions']}")
    print(f"   top diagnosis: {top['name']} ({top['confidence']}%)")
    print(f"   model predictions present: {'model_predictions' in res}")

    if res["case"]["redactions"] < 3:
        print("   FAIL: expected at least 3 PII redactions")
        ok = False
    if "Priya" in json.dumps(res["extracted"]) or "9876543210" in json.dumps(res["extracted"]):
        print("   FAIL: PII leaked into the extracted payload")
        ok = False

    print("\n2. Verify the stored text was scrubbed...")
    export = get("/api/export/training-data")
    mine = [e for e in export["examples"] if e["label"] == top["id"]]
    stored_text = mine[-1]["text"] if mine else ""
    print(f"   stored text: {stored_text[:110]}")
    for bad in ("Priya", "Sharma", "9876543210", "priya.sharma@example.com"):
        if bad in stored_text:
            print(f"   FAIL: '{bad}' survived scrubbing")
            ok = False

    print("\n3. Submit a correction...")
    fb = post("/api/feedback", {
        "case_id": case_id,
        "correct_diagnosis": "covid19",
        "rating": 4,
        "comment": "Also lost my sense of smell.",
    })
    print(f"   feedback_id={fb['feedback_id']}")

    print("\n4. Confirm the correction is picked up as a weighted training example...")
    export2 = get("/api/export/training-data")
    corrected = [e for e in export2["examples"] if e["origin"] == "feedback"]
    print(f"   corrected examples: {len(corrected)}")
    if not corrected or corrected[0]["label"] != "covid19":
        print("   FAIL: correction not reflected in training data")
        ok = False
    else:
        print(f"   label overridden to '{corrected[0]['label']}' with weight {corrected[0]['weight']}")

    print("\n5. Analyse WITHOUT consent and confirm the text is not retained...")
    res2 = post("/api/analyze", {
        "text": "I have burning when I pee and need to go very often for 2 days",
        "consent_share": False,
    })
    cid2 = res2["case"]["case_id"]
    export3 = get("/api/export/training-data")
    leaked = [e for e in export3["examples"] if "burning when I pee" in e["text"]]
    print(f"   stored_for_training={res2['case']['stored_for_training']} leaked={len(leaked)}")
    if res2["case"]["stored_for_training"] or leaked:
        print("   FAIL: non-consented text was retained")
        ok = False

    print("\n6. Trigger a retrain and wait for it...")
    before = get("/api/model/status")["model"]["meta"].get("trained_at")
    post("/api/model/train", {"backend": "tfidf", "use_seed": True})
    for _ in range(30):
        time.sleep(2)
        st = get("/api/model/status")
        after = st["model"]["meta"].get("trained_at")
        if after != before:
            break
    print(f"   trained_at: {before} -> {after}")
    print(f"   backend={st['model']['backend']} accuracy={st['model']['meta'].get('accuracy')}")
    print(f"   data stats: {json.dumps(st['data'], indent=6)[:400]}")

    print("\n7. Right-to-erasure...")
    print(f"   {delete(f'/api/data/case/{cid2}')}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
