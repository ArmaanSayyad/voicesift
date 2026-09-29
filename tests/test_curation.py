import copy, io, json
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient
from repair_bench.curation.store import Corpus, Conflict
from repair_bench.curation.detect import detect
from repair_bench.curation.models import Review
from repair_bench.curation.api import create_app

DEMO = Path(__file__).resolve().parents[1] / "fixtures/interruption-demo.jsonl"


def example():
    return json.loads(DEMO.read_text().splitlines()[0])


def save(corpus, body=None):
    return corpus.import_jsonl(json.dumps(body or example()))


def test_detection_boundaries_direction_and_missing():
    x = example()
    result, coverage = detect(x)
    assert len(result) == 1 and result[0]["overlap_s"] == 1.5
    x["turns"][2]["start_s"] = 6.0
    assert not detect(x)[0]  # user starts at agent offset
    x["turns"][2]["start_s"] = 2.0
    assert not detect(x)[0]  # simultaneous start is not agent-first
    x["turns"][2]["start_s"] = 1.9
    assert not detect(x)[0]  # agent starts over user, reverse direction
    x["turns"][2]["start_s"] = x["turns"][2]["end_s"] = None
    assert detect(x)[1]["untimed_user_turns"] == 1


def test_segmentation_does_not_create_new_interruptions():
    x = example()
    x["turns"][2]["end_s"] = 5.0
    x["turns"].append({"role": "user", "text": "more", "start_s": 5.0, "end_s": 6.5})
    r = detect(x)[0]
    assert len(r) == 1 and r[0]["overlap_s"] == 1.5
    x["turns"][1]["end_s"] = 4.5
    x["turns"].append(
        {"role": "assistant", "text": "continued", "start_s": 4.5, "end_s": 6.0}
    )
    assert detect(x)[0][0]["overlap_s"] == 1.5


def test_backchannel_still_candidate():
    x = json.loads(DEMO.read_text().splitlines()[1])
    r, _ = detect(x)
    assert len(r) == 1 and r[0]["status"] == "unreviewed_overlap_candidate"


def test_import_atomicity_dedup_and_split(tmp_path):
    c = Corpus(tmp_path)
    assert save(c)["candidates"] == 1
    assert save(c)["duplicates"] == 1
    x = example()
    x["split"] = "test"
    with pytest.raises(Conflict):
        save(c, x)
    before = c.metrics()["conversations"]
    fresh = example()
    fresh["source_group"] = "fresh"
    fresh["turns"][0]["text"] = "new request"
    with pytest.raises(Conflict):
        c.import_jsonl(json.dumps(fresh) + "\n" + json.dumps(x))
    assert c.metrics()["conversations"] == before
    with pytest.raises(ValueError):
        c.import_jsonl(json.dumps(fresh) + '\n{"bad":true}')
    assert c.metrics()["conversations"] == before


def test_review_conflict_and_export_eligibility(tmp_path):
    c = Corpus(tmp_path)
    save(c)
    cid = c.ids()[0]
    with pytest.raises(ValueError):
        c.export()
    c.review(cid, Review(label="backchannel", reviewer="test", expected_version=0))
    with pytest.raises(ValueError):
        c.export()
    with pytest.raises(Conflict):
        c.review(cid, Review(label="uncertain", reviewer="test", expected_version=0))
    with pytest.raises(ValueError):
        Review(label="backchannel", include=True, reviewer="test", expected_version=1)
    c.review(
        cid,
        Review(
            label="interruption",
            include=True,
            reviewer="test",
            note="Fixture-only annotation",
            expected_version=1,
        ),
    )
    export = c.export()
    path = c.root / "exports" / f"{export['id']}.jsonl"
    record = json.loads(path.read_text())
    assert record["detection"]["overlap_s"] == 1.5
    assert record["annotation"]["version"] == 2
    assert record["conversation"]["turns"] == c.get(cid)["conversation"]["turns"]
    snapshot = path.read_bytes()
    c.review(cid, Review(label="uncertain", reviewer="test", expected_version=2))
    assert path.read_bytes() == snapshot
    with pytest.raises(ValueError):
        c.export()


def test_untimed_retained_but_not_scored(tmp_path):
    c = Corpus(tmp_path)
    x = example()
    for t in x["turns"]:
        t["start_s"] = t["end_s"] = None
    assert save(c, x)["candidates"] == 0
    assert c.metrics()["untimed_user_turns"] == 2


def test_api_auth_import_review_audio_export(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        token = client.get("/api/curation/bootstrap").json()["token"]
        h = {"x-repair-token": token}
        assert (
            client.post(
                "/api/curation/import", json={"jsonl": DEMO.read_text()}
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/curation/import",
                json={"jsonl": DEMO.read_text()},
                headers={**h, "origin": "https://foreign.example"},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/curation/import", json={"jsonl": DEMO.read_text()}, headers=h
            ).json()["candidates"]
            == 2
        )
        c = client.get("/api/curation/candidates").json()["items"][0]
        cid = c["conversation_id"]
        audio = io.BytesIO()
        sf.write(audio, np.zeros(16000 * 9), 16000, format="WAV")
        assert (
            client.post(
                f"/api/curation/conversations/{cid}/audio", content=b"bad", headers=h
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/api/curation/conversations/{cid}/audio",
                content=audio.getvalue(),
                headers=h,
            ).status_code
            == 200
        )
        assert (
            client.post(
                f"/api/curation/conversations/{cid}/audio",
                content=audio.getvalue(),
                headers=h,
            ).status_code
            == 409
        )
        assert client.get(f"/api/curation/conversations/{cid}/audio").status_code == 200
        assert (
            client.post(
                "/api/curation/reviews/" + c["id"],
                json={
                    "label": "interruption",
                    "include": True,
                    "reviewer": "test",
                    "expected_version": 0,
                },
                headers=h,
            ).status_code
            == 200
        )
        e = client.post("/api/curation/exports", json={}, headers=h).json()
        assert client.get("/api/curation/exports/" + e["id"]).status_code == 200
        assert client.get("/api/curation/exports/unknown").status_code == 404
        path = tmp_path / "exports" / f"{e['id']}.jsonl"
        path.write_text("corrupt")
        assert client.get("/api/curation/exports/" + e["id"]).status_code == 409


def test_export_refuses_changed_source_audio(tmp_path):
    c = Corpus(tmp_path)
    save(c)
    cid = c.ids()[0]
    conversation = c.get(cid)["conversation_id"]
    with c.db() as db:
        db.execute(
            "INSERT INTO assets VALUES (?,?,?,?)",
            (conversation, "test.wav", "incorrect-hash", 10.0),
        )
    (c.root / "assets/test.wav").write_bytes(b"changed")
    c.review(
        cid,
        Review(
            label="interruption", include=True, reviewer="unit-test", expected_version=0
        ),
    )
    with pytest.raises(ValueError, match="audio hash mismatch"):
        c.export()
