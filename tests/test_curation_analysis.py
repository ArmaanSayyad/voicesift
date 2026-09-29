import io
import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from repair_bench.curation.analysis import Analysis, evidence
from repair_bench.curation.api import create_app


def setup(tmp_path, backend):
    app = create_app(tmp_path, backend=backend)
    client = TestClient(app)
    client.headers["x-repair-token"] = client.get("/api/curation/bootstrap").json()[
        "token"
    ]
    body = {
        "id": "test",
        "source_group": "test",
        "provenance": "Constructed test",
        "turns": [
            {
                "role": "assistant",
                "text": "Here are options",
                "start_s": 0.0,
                "end_s": 3.0,
            },
            {"role": "user", "text": "Wait please", "start_s": 1.0, "end_s": 2.0},
        ],
    }
    assert (
        client.post(
            "/api/curation/import", json={"jsonl": json.dumps(body)}
        ).status_code
        == 200
    )
    c = client.get("/api/curation/candidates").json()["items"][0]
    b = io.BytesIO()
    sf.write(b, np.zeros((64000, 2)), 16000, format="WAV")
    assert (
        client.post(
            f"/api/curation/conversations/{c['conversation_id']}/audio",
            content=b.getvalue(),
        ).status_code
        == 200
    )
    return app, client, c


def wait(client):
    for _ in range(200):
        job = client.get("/api/curation/jobs").json()["items"][0]
        if job["status"] == "complete":
            return job
        time.sleep(0.01)
    raise AssertionError("Job did not finish")


def test_analysis_cache_review_export_and_clip(tmp_path):
    calls = []

    def backend(body):
        calls.append(body)
        return {
            "answer": {
                "intent": "take_floor",
                "agent_outcome": "unknown",
                "evidence_note": "User takes floor",
            },
            "usage": {},
        }

    app, client, c = setup(tmp_path, backend)
    for n in range(2):
        assert (
            client.post(
                "/api/curation/analyze", json={"candidate_ids": [c["id"]]}
            ).status_code
            == 200
        )
        assert wait(client)["errors"] == 0
    assert len(calls) == 1
    result = client.get("/api/curation/candidates").json()["items"][0]
    assert result["analysis"]["cached"] and result["review"] is None
    assert client.post("/api/curation/exports").status_code == 422
    clip = client.get(f"/api/curation/candidates/{c['id']}/clip")
    assert sf.info(io.BytesIO(clip.content)).samplerate == 16000
    assert (
        client.post(
            "/api/curation/reviews/" + c["id"],
            json={
                "label": "interruption",
                "include": True,
                "agent_outcome": "unknown",
                "reviewer": "automated-test-not-human",
                "expected_version": 0,
            },
        ).status_code
        == 200
    )
    export = client.post("/api/curation/exports").json()
    record = json.loads(client.get("/api/curation/exports/" + export["id"]).text)
    assert (
        record["schema_version"] == 2
        and record["model_suggestion"]["answer"]["intent"] == "take_floor"
    )
    assert record["annotation"]["agent_outcome"] == "unknown"
    import zipfile

    bundle = client.get("/api/curation/exports/" + export["id"] + "/bundle")
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        assert "annotations.jsonl" in archive.namelist()
        assert len(json.loads(archive.read("clips.json"))) == 1
    asset = app.state.corpus.root / "assets" / result["audio"]["name"]
    asset.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        evidence(app.state.corpus, c["id"])
    app.state.analysis.pool.shutdown()


def test_failures_retry_and_concurrency(tmp_path):
    gate = threading.Event()

    def bad(body):
        gate.wait(1)
        raise RuntimeError("private provider content")

    app, client, c = setup(tmp_path, bad)
    assert (
        client.post(
            "/api/curation/analyze", json={"candidate_ids": [c["id"], c["id"]]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/curation/analyze", json={"candidate_ids": [c["id"]]}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/curation/analyze", json={"candidate_ids": [c["id"]]}
        ).status_code
        == 409
    )
    gate.set()
    assert wait(client)["errors"] == 1
    result = client.get("/api/curation/candidates").json()["items"][0]
    assert result["analysis"]["error"] == "RuntimeError" and result["review"] is None
    assert (
        client.post(
            "/api/curation/analyze", json={"candidate_ids": [c["id"]]}
        ).status_code
        == 200
    )
    assert wait(client)["errors"] == 1
    app.state.analysis.pool.shutdown()


def test_restart_marks_abandoned_jobs(tmp_path):
    app, _client, _c = setup(tmp_path, lambda _: None)
    with app.state.corpus.db() as db:
        db.execute(
            "INSERT INTO analysis_jobs VALUES ('abandoned','running',1,0,0,'now','[]')"
        )
    restarted = Analysis(app.state.corpus, lambda _: None)
    assert restarted.jobs()[0]["status"] == "interrupted"
    restarted.pool.shutdown()
    app.state.analysis.pool.shutdown()


def test_missing_audio_and_truncated_clip_are_explicit(tmp_path):
    from repair_bench.curation.store import Corpus

    corpus = Corpus(tmp_path)
    body = {
        "id": "long",
        "source_group": "long",
        "provenance": "test",
        "turns": [
            {"role": "assistant", "text": "agent", "start_s": 0.0, "end_s": 60.0},
            {"role": "user", "text": "user", "start_s": 1.0, "end_s": 59.0},
        ],
    }
    corpus.import_jsonl(json.dumps(body))
    cid = corpus.ids()[0]
    with pytest.raises(ValueError, match="Attach source WAV"):
        evidence(corpus, cid)
    import hashlib

    data = io.BytesIO()
    sf.write(data, np.zeros((60 * 8000, 2)), 8000, format="WAV")
    path = corpus.root / "assets" / "long.wav"
    path.write_bytes(data.getvalue())
    c = corpus.get(cid)
    with corpus.db() as db:
        db.execute(
            "INSERT INTO assets VALUES (?,?,?,?)",
            (
                c["conversation_id"],
                "long.wav",
                hashlib.sha256(data.getvalue()).hexdigest(),
                60,
            ),
        )
    packet, audio, meta = evidence(corpus, cid)
    assert packet["target_truncated"]
    info = sf.info(io.BytesIO(audio))
    assert info.duration == 30 and info.channels == 2 and info.samplerate == 16000
    assert meta["source_audio_sha256"] != meta["clip_sha256"]
