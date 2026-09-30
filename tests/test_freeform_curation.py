"""End-to-end contract tests; controlled answers are not accuracy measurements."""

import io
import json
import time
import zipfile
from urllib.parse import quote

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from repair_bench.curation import objectives
from repair_bench.curation.api import create_app
from repair_bench.curation.runs import Runs
from repair_bench.curation.sources import REQUIREMENT


def bundle(tmp_path, count=3, seconds=1):
    audio = io.BytesIO()
    sf.write(audio, np.zeros(int(seconds * 16000)), 16000, format="WAV")
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w") as z:
        rows = [
            {"id": str(i), "audio": f"{i}.wav", "gold_label": "MUST_NOT_REACH_MODEL"}
            for i in range(count)
        ]
        z.writestr("dataset.jsonl", "\n".join(map(json.dumps, rows)))
        for i in range(count):
            # Distinct audio gives each recording a separate request fingerprint.
            a = io.BytesIO()
            sf.write(a, np.full(int(seconds * 16000), i / 10), 16000, format="WAV")
            z.writestr(f"{i}.wav", a.getvalue())
    return path


def finish(runs, rid, states=None):
    states = states or {
        "completed",
        "completed_with_errors",
        "failed",
        "needs_clarification",
        "paused",
        "ready",
    }
    for _ in range(500):
        row = runs.get(rid)
        if row["status"] in states:
            return row
        time.sleep(0.01)
    raise AssertionError(runs.get(rid))


def plan(supported=True):
    return {
        "answer": {
            "supported": supported,
            "summary": "Audible laughter",
            "criteria": ["Clear audible laughter"] if supported else [],
            "reason": "" if supported else "Specify an audible property.",
        }
    }


def is_plan(body):
    return "supported" in body["generationConfig"]["responseSchema"]["properties"]


def test_freeform_upload_review_export_and_cache(tmp_path):
    calls = []
    labels = iter(["yes", "no", "unclear"])

    def backend(body):
        assert "MUST_NOT_REACH_MODEL" not in json.dumps(body)
        calls.append(body)
        return (
            plan()
            if is_plan(body)
            else {
                "answer": {
                    "match": next(labels),
                    "evidence_note": "Controlled result, not an acoustic judgment.",
                }
            }
        )

    app = create_app(tmp_path / "app", backend=backend)
    with TestClient(app) as client:
        client.headers["x-repair-token"] = client.get("/api/curation/bootstrap").json()[
            "token"
        ]
        objective = "laughter but no coughing & no music?"
        r = client.post(
            "/api/curation/runs/upload?preflight=true&requirement=" + quote(objective),
            content=bundle(tmp_path).read_bytes(),
        ).json()
        runs = app.state.runs
        ready = finish(runs, r["id"])
        assert ready["status"] == "ready" and ready["candidates"] == 3 and calls == []
        base = "/api/curation/runs/" + r["id"]
        assert client.post(base + "/resume").status_code == 200
        done = finish(runs, r["id"])
        assert done["status"] == "completed" and done["unresolved"] == 1
        assert done["selected_conversations"] == 1 and done["requirement"] == objective
        detail = client.get(base + "?view=all").json()
        assert detail["counts"] == {
            "selected": 1,
            "not_selected": 1,
            "unresolved": 1,
            "not_proposed": 0,
        }
        assert all(i["turns"] == [] for i in detail["items"])
        for item in detail["items"]:
            assert (
                client.get(base + "/events/" + item["id"] + "/audio").status_code == 200
            )
        event = detail["items"][1]
        assert (
            client.put(
                base + "/reviews/" + event["id"],
                json={"decision": "keep", "revision": 0},
            ).status_code
            == 200
        )
        export = client.post(base + "/reviewed-exports", json={"revision": 1}).json()
        with zipfile.ZipFile(
            io.BytesIO(client.get(base + "/reviewed-exports/" + export["id"]).content)
        ) as z:
            manifest = json.loads(z.read("manifest.json"))
            assert (
                manifest["requirement"] == objective
                and manifest["objective_plan"]["supported"]
            )
            assert len(z.read("dataset.jsonl").splitlines()) == 1
        rerun = client.post(base + "/rerun").json()
        client.post("/api/curation/runs/" + rerun["id"] + "/resume")
        assert finish(runs, rerun["id"])["status"] == "completed"
        # A rerun preserves its interpretation and reuses all three audio decisions.
        assert len(calls) == 4
        with zipfile.ZipFile(runs.download(r["id"])) as z:
            manifest = json.loads(z.read("manifest.json"))
            assert manifest["policy_version"] == objectives.VERSION
            assert "successful-interruption" not in manifest["selection"]
            assert len(z.read("all-decisions.jsonl").splitlines()) == 3


def test_unsupported_objective_stops_before_audio_calls(tmp_path):
    calls = []

    def backend(body):
        calls.append(body)
        assert is_plan(body)
        return plan(False)

    with RunsContext(tmp_path, backend) as runs:
        r = runs.submit("Find people wearing blue", upload=bundle(tmp_path))
        done = finish(runs, r["id"])
        assert done["status"] == "needs_clarification" and not done["download_ready"]
        assert len(calls) == 1 and done["objective_plan"]["reason"]


class RunsContext:
    def __init__(self, tmp_path, backend):
        self.runs = Runs(tmp_path / "runs", backend=backend)

    def __enter__(self):
        return self.runs

    def __exit__(self, *args):
        self.runs.shutdown()


def test_invalid_judgment_is_retryable_and_not_negative(tmp_path):
    invalid = [True]

    def backend(body):
        if is_plan(body):
            return plan()
        return {
            "answer": {
                "match": "perhaps" if invalid[0] else "yes",
                "evidence_note": "audible event",
            }
        }

    with RunsContext(tmp_path, backend) as runs:
        r = runs.submit("laughter", upload=bundle(tmp_path, count=1))
        done = finish(runs, r["id"])
        assert (
            done["status"] == "completed_with_errors" and done["selected_events"] == 0
        )
        assert runs.detail(r["id"], view="unresolved")["total"] == 1
        invalid[0] = False
        runs.resume(r["id"])
        assert finish(runs, r["id"])["selected_events"] == 1


def test_full_audio_limit_fails_before_spend(tmp_path):
    def never(body):
        raise AssertionError("must not call model")

    with RunsContext(tmp_path, never) as runs:
        r = runs.submit("laughter", upload=bundle(tmp_path, count=1, seconds=301))
        done = finish(runs, r["id"])
        assert done["status"] == "failed" and "5 minutes" in done["message"]


def test_default_without_transcripts_uses_audio_path(tmp_path):
    def backend(body):
        return (
            plan()
            if is_plan(body)
            else {
                "answer": {
                    "match": "unclear",
                    "evidence_note": "Cannot identify an interruption.",
                }
            }
        )

    with RunsContext(tmp_path, backend) as runs:
        r = runs.submit(REQUIREMENT, upload=bundle(tmp_path, count=1))
        assert finish(runs, r["id"])["mode"] == "audio"


@pytest.mark.parametrize("value", ["", "  ", "x" * 4001, None, 42])
def test_objective_validation(value):
    with pytest.raises(ValueError):
        objectives.requirement(value)


def test_objective_is_part_of_prediction_cache(tmp_path):
    calls = []

    def backend(body):
        calls.append(body)
        return {"answer": {"match": "no", "evidence_note": "Absent"}}

    with RunsContext(tmp_path, backend) as runs:
        runs.predict(None, b"audio", "laughter", plan()["answer"])
        runs.predict(None, b"audio", "coughing", plan()["answer"])
        runs.predict(None, b"audio", "laughter", plan()["answer"])
        assert len(calls) == 2


def test_inline_audio_limit_fails_before_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(objectives, "MAX_AUDIO_BYTES", 10)

    def never(body):
        raise AssertionError("must not call model")

    with RunsContext(tmp_path, never) as runs:
        row = runs.submit("laughter", upload=bundle(tmp_path, count=1))
        done = finish(runs, row["id"])
        assert done["status"] == "failed" and "14 MB" in done["message"]


def test_planner_rate_limit_is_resumable(tmp_path):
    from urllib.error import HTTPError

    limited = [True]

    def backend(body):
        if limited[0]:
            raise HTTPError("provider", 429, "private request must not leak", {}, None)
        return (
            plan()
            if is_plan(body)
            else {"answer": {"match": "no", "evidence_note": "Absent"}}
        )

    with RunsContext(tmp_path, backend) as runs:
        row = runs.submit("laughter", upload=bundle(tmp_path, count=1))
        done = finish(runs, row["id"])
        assert done["status"] == "paused" and "rate limit" in done["message"]
        assert "private" not in done["message"]
        limited[0] = False
        runs.resume(row["id"])
        assert finish(runs, row["id"])["status"] == "completed"


def test_freeform_huggingface_source_preserves_objective(tmp_path):
    from repair_bench.curation.sources import unpack

    source = unpack(bundle(tmp_path, count=1), tmp_path / "source")
    calls = []

    def backend(body):
        calls.append(body)
        return (
            plan()
            if is_plan(body)
            else {"answer": {"match": "yes", "evidence_note": "Controlled match"}}
        )

    app = create_app(tmp_path / "app", backend=backend)
    app.state.runs.downloader = lambda *args: (
        source,
        {"adapter": "normalized-jsonl", "revision": "pinned-test-revision"},
        [],
    )
    with TestClient(app) as client:
        client.headers["x-repair-token"] = client.get("/api/curation/bootstrap").json()[
            "token"
        ]
        objective = "A spoken correction or cancellation"
        response = client.post(
            "/api/curation/runs/huggingface",
            json={
                "url": "https://huggingface.co/datasets/test/audio",
                "requirement": objective,
            },
        )
        assert response.status_code == 200
        row = response.json()
        assert finish(app.state.runs, row["id"])["selected_conversations"] == 1
        assert calls[0]["contents"][0]["parts"][0]["text"] == objective
        with zipfile.ZipFile(app.state.runs.download(row["id"])) as z:
            manifest = json.loads(z.read("manifest.json"))
            assert manifest["requirement"] == objective
            assert manifest["source_provenance"]["revision"] == "pinned-test-revision"


def test_freeform_does_not_gate_timed_audio_on_overlap(tmp_path):
    from test_dataset_runs import bundle as conversation_bundle

    turns = [
        {"role": "assistant", "text": "Hello", "start_s": 0, "end_s": 1},
        {"role": "user", "text": "Cancel that", "start_s": 2, "end_s": 3},
    ]
    calls = []

    def backend(body):
        calls.append(body)
        return (
            plan()
            if is_plan(body)
            else {"answer": {"match": "yes", "evidence_note": "Controlled match"}}
        )

    with RunsContext(tmp_path, backend) as runs:
        row = runs.submit(
            "Conversations with a cancellation",
            upload=conversation_bundle(tmp_path, turns=turns),
        )
        done = finish(runs, row["id"])
        assert done["selected_conversations"] == 1 and done["mode"] == "audio"
        assert len(calls) == 2
