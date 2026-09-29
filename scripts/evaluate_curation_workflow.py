"""Actual API + real Gemini on constructed dialogue fixtures with fresh SID clips."""

import io
import json
import time
import zipfile
from pathlib import Path

from evaluate_interruption_laya import score
from fastapi.testclient import TestClient

from repair_bench.curation.api import create_app

OUT = Path("artifacts/workflow-eval")
manifest = json.loads((OUT / "manifest.json").read_text())
app = create_app(OUT / "corpus")
client = TestClient(app)
client.headers["x-repair-token"] = client.get("/api/curation/bootstrap").json()["token"]
response = client.post(
    "/api/curation/import",
    json={"jsonl": "\n".join(json.dumps(r["conversation"]) for r in manifest)},
)
assert response.status_code == 200, response.text
candidates = client.get("/api/curation/candidates").json()["items"]
lookup = {r["conversation"]["id"]: r for r in manifest}
assert len(candidates) == 40
assert all(not lookup[c["conversation"]["id"]]["control"] for c in candidates)
for c in candidates:
    if not c["audio"]:
        r = client.post(
            "/api/curation/conversations/" + c["conversation_id"] + "/audio",
            content=Path(lookup[c["conversation"]["id"]]["audio"]).read_bytes(),
        )
        assert r.status_code == 200, r.text
assert client.post("/api/curation/exports").status_code == 422
response = client.post(
    "/api/curation/analyze", json={"candidate_ids": [c["id"] for c in candidates]}
)
assert response.status_code == 200, response.text
while True:
    job = client.get("/api/curation/jobs").json()["items"][0]
    if job["status"] == "complete":
        break
    time.sleep(1)
print("job", job["completed"], job["errors"], flush=True)
results = []
for c in client.get("/api/curation/candidates").json()["items"]:
    r = lookup[c["conversation"]["id"]]
    a = c["analysis"]
    answer = a.get("answer", {})
    results.append(
        {
            "id": c["conversation"]["id"],
            "positive": r["positive"],
            "prediction": answer.get("intent") == "take_floor",
            "intent": answer.get("intent"),
            "agent_outcome": answer.get("agent_outcome"),
            "status": a["status"],
            "source_audio_sha256": a.get("evidence", {}).get("source_audio_sha256"),
            "request_sha256": a.get("request_sha256"),
            "usage": a.get("usage", {}),
            "seconds": a.get("seconds"),
        }
    )
    assert c["review"] is None
    # Automated fixture annotations test export mechanics, not human-reviewed truth.
    label = "interruption" if r["positive"] else "backchannel"
    response = client.post(
        "/api/curation/reviews/" + c["id"],
        json={
            "label": label,
            "include": r["positive"],
            "agent_outcome": "unknown",
            "reviewer": "automated-fixture-test",
            "note": "Constructed fixture label from source intent; NOT human reviewed, NOT natural dialogue ground truth.",
            "expected_version": 0,
        },
    )
    assert response.status_code == 200, response.text
export = client.post("/api/curation/exports").json()
assert export["records"] == 20
bundle = client.get("/api/curation/exports/" + export["id"] + "/bundle")
assert bundle.status_code == 200
with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
    assert len(json.loads(archive.read("clips.json"))) == 20
summary = {
    "scope": "40 new SID source clips (20 positive,20 negative) overlapped with fixed Kokoro agent speech, plus 10 non-overlap controls. Constructed audio/timelines; source semantic labels are proxies, not human annotations of the new conversations. Reference user transcripts supplied, not ASR. Not a natural-conversation accuracy estimate.",
    "input_conversations": 50,
    "overlap_candidates": 40,
    "nonoverlap_controls_excluded": 10,
    "candidate_errors": 0,
    "analysis_errors": job["errors"],
    "intent_proxy_metrics": score(results, {r["id"]: r["prediction"] for r in results}),
    "unclear": sum(r["intent"] == "unclear" for r in results),
    "automated_fixture_exports": 20,
    "zip_clips": 20,
    "model_prompt_changed_from_prior_ablation": True,
    "estimated_api_usd": sum(
        (
            r.get("usage", {}).get("promptTokenCount", 0) * 0.75
            + (
                r.get("usage", {}).get("candidatesTokenCount", 0)
                + r.get("usage", {}).get("thoughtsTokenCount", 0)
            )
            * 3.75
        )
        / 1e6
        for r in results
    ),
}
(OUT / "results.json").write_text(
    json.dumps({"summary": summary, "predictions": results}, indent=2)
)
print(json.dumps(summary, indent=2), flush=True)
app.state.analysis.pool.shutdown()
