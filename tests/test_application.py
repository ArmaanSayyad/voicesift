import copy
import json
import threading
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from pydantic import ValidationError
from repair_bench.api import create_app
from repair_bench.contracts import Plan, verbalize
from repair_bench.service import Coordinator
from repair_bench.store import Store


class FakeWorker:
    def __init__(self):
        self.calls = []

    def close(self):
        pass

    def request(self, op, **kwargs):
        self.calls.append((op, kwargs))
        if op == "prepare":
            return {"ready": True}
        if op == "asr":
            return {
                "text": "Plan Friday at seven for two."
                if "input-0" in kwargs["path"]
                else "Actually Saturday. Keep everything else.",
                "input_duration_s": 1,
            }
        sf.write(kwargs["path"], np.zeros(2400), 24000)
        return {"sample_rate": 24000, "samples": 2400, "duration_s": 0.1}


@pytest.fixture
def manager(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    a = repo / "artifacts"
    a.mkdir(parents=True)
    (a / "models.json").write_text(
        json.dumps(
            {
                "parakeet": {"path": "unused", "revision": "fake"},
                "kokoro": {"path": "unused", "revision": "fake"},
            }
        )
    )
    for i in range(2):
        sf.write(a / f"phrase-{i}.wav", np.zeros(2400), 24000)
    requests = []

    def chat(path, payload=None):
        if path == "/api/tags":
            return {"models": [{"name": "qwen3.5:9b", "digest": "fake"}]}
        requests.append(copy.deepcopy(payload))
        value = {
            "day": "Friday" if len(requests) == 1 else "Saturday",
            "time": "19:00",
            "party_size": 2,
            "status": "proposed",
        }
        return {"message": {"content": json.dumps(value)}}

    monkeypatch.setattr(
        "repair_bench.service.subprocess.check_output", lambda *a, **kw: "test"
    )
    m = Coordinator(repo, worker=FakeWorker(), chat=chat)
    m.requests = requests
    yield m
    m.close()


def test_complete_run_and_no_future_oracle_leak(manager):
    r = manager.store.create("dinner-date-dev")
    manager.execute(r)
    detail = manager.store.detail(r)
    assert detail["status"] == "complete"
    assert detail["manifest"]["score"]["value"] is True
    assert manager.store.verify(r)["verified"]
    first = json.dumps(manager.requests[0])
    assert "Actually Saturday" not in first and "expected" not in first
    assert "score" not in json.dumps(manager.requests)
    assert len(detail["manifest"]["turns"]) == 2
    assert not any(e["kind"] == "audio.rendered" for e in manager.store.events(r))
    path = manager.store.folder(r) / "response-0.wav"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        manager.store.verify(r)


def test_bad_model_schema_is_failed_process_not_zero_score(manager):
    original = manager.chat
    manager.chat = (
        lambda path, payload=None: original(path, payload)
        if path == "/api/tags"
        else {"message": {"content": '{"party_size":true}'}}
    )
    run = manager.store.create("dinner-date-dev")
    manager.execute(run)
    result = manager.store.detail(run)
    assert result["status"] == "failed" and "score" not in result["manifest"]
    assert manager.store.events(run)[-1]["kind"] == "run.failed"
    assert manager.active is None


def test_startup_marks_incomplete_attempts_interrupted(tmp_path):
    s = Store(tmp_path)
    r = s.create("x")
    s.update(r, "running")
    recovered = Store(tmp_path)
    assert recovered.get(r)["status"] == "interrupted"


def test_same_origin_token_and_asset_access(manager):
    with TestClient(create_app(manager)) as client:
        assert client.post("/api/v1/runs", json={}).status_code == 403
        token = client.get("/api/v1/bootstrap").json()["token"]
        headers = {"x-repair-token": token, "origin": "https://foreign.example"}
        assert client.post("/api/v1/runs", json={}, headers=headers).status_code == 403
        assert (
            client.get("/api/v1/runs", headers={"host": "foreign.example"}).status_code
            == 400
        )
        assert client.get("/api/v1/runs/unknown").status_code == 404
        run = manager.store.create("dinner-date-dev")
        manager.execute(run)
        assert client.get(f"/api/v1/runs/{run}/audio/input-0.wav").status_code == 200
        assert client.get(f"/api/v1/runs/{run}/audio/index.sqlite").status_code == 404
        assert client.get(f"/api/v1/runs/{run}/verify").json()["verified"]
        assert (
            client.post(
                "/api/v1/runs",
                json={"scenario_id": "untrusted"},
                headers={"x-repair-token": token},
            ).status_code
            == 422
        )


def test_one_active_run(manager):
    entered = threading.Event()
    release = threading.Event()
    original = manager.worker.request

    def request(op, **kwargs):
        if op == "prepare":
            entered.set()
            release.wait(5)
        return original(op, **kwargs)

    manager.worker.request = request
    manager.start("dinner-date-dev")
    assert entered.wait(3)
    try:
        with pytest.raises(RuntimeError, match="already active"):
            manager.start("dinner-date-dev")
    finally:
        release.set()
        manager.pool.shutdown(wait=True)


def test_verbalizer_and_strict_schema():
    p = Plan(day="Saturday", time="19:05", party_size=2, status="proposed")
    assert "seven oh five P M" in verbalize(p)
    assert "19:05" not in verbalize(p)
    assert "cancelled" in verbalize(p.model_copy(update={"status": "cancelled"}))
    assert "clarify" in verbalize(p.model_copy(update={"day": None}))
    with pytest.raises(ValidationError):
        Plan(day="Saturday", time="25:00", party_size=2, status="proposed")
    with pytest.raises(ValidationError):
        Plan(day="Saturday", time="19:00", party_size=True, status="proposed")
