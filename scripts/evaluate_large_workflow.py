"""Frozen production classifier on a larger constructed corpus; four isolated stores."""

import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_interruption_laya import score
from fastapi.testclient import TestClient

from repair_bench.curation.analysis import MODEL, PROMPT, SCHEMA
from repair_bench.curation.api import create_app
from repair_bench.curation.store import Corpus

OUT = Path("artifacts/large-workflow-eval")


def run_shard(item):
    shard, rows = item
    app = create_app(OUT / f"corpus-{shard}")
    client = TestClient(app)
    client.headers["x-repair-token"] = client.get("/api/curation/bootstrap").json()[
        "token"
    ]
    response = client.post(
        "/api/curation/import",
        json={"jsonl": "\n".join(json.dumps(r["conversation"]) for r in rows)},
    )
    assert response.status_code == 200, response.text
    candidates = client.get("/api/curation/candidates").json()["items"]
    lookup = {r["conversation"]["id"]: r for r in rows}
    expected = {r["conversation"]["id"] for r in rows if not r["control"]}
    assert {c["conversation"]["id"] for c in candidates} == expected
    for c in candidates:
        if not c["audio"]:
            response = client.post(
                "/api/curation/conversations/" + c["conversation_id"] + "/audio",
                content=Path(lookup[c["conversation"]["id"]]["audio"]).read_bytes(),
            )
            assert response.status_code == 200, response.text
    pending = [c["id"] for c in candidates if c["analysis"] is None]
    for offset in range(0, len(pending), 50):
        response = client.post(
            "/api/curation/analyze",
            json={"candidate_ids": pending[offset : offset + 50]},
        )
        assert response.status_code == 200, response.text
        while True:
            job = client.get("/api/curation/jobs").json()["items"][0]
            if job["status"] == "complete":
                break
            time.sleep(1)
        print(
            "shard",
            shard,
            "batch",
            offset // 50,
            "completed",
            job["completed"],
            "errors",
            job["errors"],
            flush=True,
        )
    result = []
    for c in client.get("/api/curation/candidates").json()["items"]:
        source = lookup[c["conversation"]["id"]]
        a = c["analysis"]
        answer = a.get("answer", {})
        result.append(
            {
                "id": c["conversation"]["id"],
                "positive": source["positive"],
                "prediction": answer.get("intent") == "take_floor",
                "intent": answer.get("intent"),
                "status": a["status"],
                "request_sha256": a.get("request_sha256"),
                "source_audio_sha256": a.get("evidence", {}).get("source_audio_sha256"),
                "truncated": a.get("evidence", {}).get("target_truncated"),
                "seconds": a.get("seconds"),
                "usage": a.get("usage", {}),
                "control": False,
            }
        )
        assert c["review"] is None
    with app.state.corpus.db() as db:
        accepted = {r[0] for r in db.execute("SELECT external_id FROM conversations")}
    for r in rows:
        if r["control"] and r["conversation"]["id"] in accepted:
            result.append(
                {
                    "id": r["conversation"]["id"],
                    "positive": False,
                    "prediction": False,
                    "intent": None,
                    "status": "excluded_by_timing",
                    "control": True,
                }
            )
    assert client.post("/api/curation/exports").status_code == 422
    app.state.analysis.pool.shutdown()
    (OUT / f"results-{shard}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert len(manifest) == 400
    canonical_corpus = Corpus(OUT / "corpus-all")
    for offset in range(0, len(manifest), 100):
        canonical_corpus.import_jsonl(
            "\n".join(
                json.dumps(r["conversation"]) for r in manifest[offset : offset + 100]
            )
        )
    with canonical_corpus.db() as db:
        accepted_ids = {
            r[0] for r in db.execute("SELECT external_id FROM conversations")
        }
    assert len(canonical_corpus.ids()) == 300
    protocol = {
        "model": MODEL,
        "prompt": PROMPT,
        "schema": SCHEMA,
        "generation": {"temperature": 0, "maxOutputTokens": 1024},
        "production_analysis_sha256": hashlib.sha256(
            Path("src/repair_bench/curation/analysis.py").read_bytes()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (OUT / "manifest.json").read_bytes()
        ).hexdigest(),
        "scope": "400 constructed conversations; 300 source clips new to Gemini evaluation, reference transcripts, proxy source intent labels, no natural dialogue ground truth. Four isolated app stores, production pipeline unchanged. No automatic retry, no human reviews manufactured.",
        "preparation": json.loads((OUT / "protocol.json").read_text()),
    }
    pp = OUT / "frozen-protocol.json"
    if pp.exists():
        assert json.loads(pp.read_text()) == protocol, "Protocol changed"
    pp.write_text(json.dumps(protocol, indent=2))
    random.Random(20261003).shuffle(manifest)
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for part in pool.map(run_shard, enumerate([manifest[i::4] for i in range(4)])):
            results.extend(part)
    results = [r for r in results if not r["control"]]
    results.extend(
        {
            "id": r["conversation"]["id"],
            "positive": False,
            "prediction": False,
            "intent": None,
            "status": "excluded_by_timing",
            "control": True,
        }
        for r in manifest
        if r["control"] and r["conversation"]["id"] in accepted_ids
    )
    assert {r["id"] for r in results} == accepted_ids
    assert len({r["id"] for r in results}) == len(results)
    assert sum(not r["control"] for r in results) == 300
    predictions = {r["id"]: r["prediction"] for r in results}
    overlap = [r for r in results if not r["control"]]
    lookup = {r["conversation"]["id"]: r for r in manifest}
    baseline = {
        r["id"]: not r["control"]
        and len(
            re.findall(
                r"\b[\w']+\b", lookup[r["id"]]["conversation"]["turns"][1]["text"]
            )
        )
        >= 4
        for r in results
    }
    retained = {
        r["id"]: not r["control"]
        and (r["status"] == "error" or r["intent"] in ("take_floor", "unclear"))
        for r in results
    }
    summary = {
        "submitted_conversations": 400,
        "conversations": len(results),
        "duplicates_skipped": 400 - len(results),
        "source_clips": 300,
        "labeled_positive": 100,
        "labeled_negative": len(results) - 100,
        "candidates": len(overlap),
        "excluded_controls": sum(r["control"] for r in results),
        "full_corpus_selection": score(results, predictions),
        "overlap_only_selection": score(overlap, predictions),
        "four_word_baseline": score(results, baseline),
        "retained_for_review": score(results, retained),
        "unclear_positive": sum(
            r["positive"] and r["intent"] == "unclear" for r in overlap
        ),
        "unclear_negative": sum(
            not r["positive"] and r["intent"] == "unclear" for r in overlap
        ),
        "errors": sum(r["status"] == "error" for r in overlap),
        "truncated": sum(bool(r.get("truncated")) for r in overlap),
        "estimated_usd": sum(
            (
                r.get("usage", {}).get("promptTokenCount", 0) * 0.75
                + (
                    r.get("usage", {}).get("candidatesTokenCount", 0)
                    + r.get("usage", {}).get("thoughtsTokenCount", 0)
                )
                * 3.75
            )
            / 1e6
            for r in overlap
        ),
        "scope": protocol["scope"],
    }
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "predictions": results}, indent=2)
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
