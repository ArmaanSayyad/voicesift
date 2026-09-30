"""Behavioral tests for recovery, review/export correctness and source preflight."""

import io
import json
import threading
import time
import urllib.error
import zipfile

import pytest
from fastapi.testclient import TestClient
from test_dataset_runs import backend, bundle

from repair_bench.curation.api import create_app
from repair_bench.curation.runs import Runs
from repair_bench.curation.sources import REQUIREMENT
from repair_bench.curation.store import Conflict


def finish(manager, rid):
    for _ in range(500):
        row = manager.get(rid)
        if row["status"] in {
            "ready",
            "completed",
            "completed_with_errors",
            "paused",
            "failed",
            "interrupted",
        }:
            return row
        time.sleep(0.01)
    raise AssertionError(manager.get(rid))


def many_turns(n=8):
    return [{"role": "assistant", "text": "Options", "start_s": 0, "end_s": 3}] + [
        {
            "role": "user",
            "text": f"Wait {i}",
            "start_s": 1 + i / 10,
            "end_s": 1.05 + i / 10,
        }
        for i in range(n)
    ]


def make_run(tmp_path, judge=backend, **kwargs):
    manager = Runs(tmp_path / "runs", backend=judge)
    row = manager.submit(REQUIREMENT, upload=bundle(tmp_path, **kwargs))
    assert finish(manager, row["id"])["status"] != "failed"
    return manager, row["id"]


def test_preflight_has_no_model_calls_and_start_uses_validated_snapshot(tmp_path):
    calls = []
    manager = Runs(tmp_path / "runs", backend=lambda b: calls.append(b) or backend(b))
    row = manager.submit(REQUIREMENT, upload=bundle(tmp_path), preflight=True)
    checked = finish(manager, row["id"])
    assert checked["status"] == "ready" and not calls
    assert checked["coverage"]["candidate_events"] == 1
    assert checked["coverage"]["timed_turns"] == 1
    # No re-extraction/download needed after preflight.
    (manager.folder(row["id"]) / "upload.zip").unlink()
    manager.resume(row["id"])
    assert finish(manager, row["id"])["status"] == "completed" and len(calls) == 1


def test_reviewed_export_only_kept_events_and_undo_preserves_original(tmp_path):
    manager, rid = make_run(tmp_path, turns=many_turns(3))
    original = manager.download(rid).read_bytes()
    events = manager.detail(rid)["items"]
    rev = 0
    for event, decision in zip(events, ["keep", "exclude", "unsure"]):
        rev = manager.review_event(rid, event["id"], decision, rev)["revision"]
    exported = manager.reviewed_export(rid, rev)
    path = manager.reviewed_download(rid, exported["id"])
    frozen = path.read_bytes()
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        labels = [
            json.loads(line) for line in z.read("selected-events.jsonl").splitlines()
        ]
        assert [e["id"] for e in labels] == [events[0]["id"]]
        assert (
            "answer" not in labels[0]
            and labels[0]["label_status"] == "human_reviewed_keep"
        )
        data = json.loads(z.read("dataset.jsonl"))
        assert data["curation"]["events"] == [events[0]["id"]]
        assert "all-decisions.jsonl" not in z.namelist()
        assert "source-notices/LICENSE" in z.namelist()
        assert len([n for n in z.namelist() if n.startswith("clips/")]) == 1
    assert manager.reviewed_export(rid, rev)["id"] == exported["id"]
    assert manager.download(rid).read_bytes() == original
    with pytest.raises(Conflict):
        manager.review_event(rid, events[0]["id"], "exclude", 0)
    with pytest.raises(Conflict):
        manager.reviewed_export(rid, 0)
    manager.review_event(rid, events[0]["id"], "unreviewed", rev)
    empty = manager.reviewed_export(rid, rev + 1)
    with zipfile.ZipFile(manager.reviewed_download(rid, empty["id"])) as z:
        assert z.read("dataset.jsonl") == b""
    assert manager.reviewed_download(rid, exported["id"]).read_bytes() == frozen
    restored = Runs(manager.root, backend=backend)
    assert restored.reviews(rid)["revision"] == rev + 1
    assert len(restored.reviewed_versions(rid)) == 2
    path.write_bytes(b"corrupted")
    with pytest.raises(Conflict):
        restored.reviewed_download(rid, exported["id"])


def test_rejected_and_unproposed_are_inspectable_and_can_be_kept(tmp_path):
    def negative(body):
        answer = backend(body)
        answer["answer"].update(
            intent="backchannel", interruption_result="not_applicable"
        )
        return answer

    turns = many_turns(1) + [
        {"role": "user", "text": "Later", "start_s": 4, "end_s": 4.5}
    ]
    manager, rid = make_run(tmp_path, negative, turns=turns)
    rejected = manager.detail(rid, view="not_selected")
    unproposed = manager.detail(rid, view="not_proposed")
    assert rejected["total"] == unproposed["total"] == 1
    assert manager.detail(rid)["total"] == 0
    event = unproposed["items"][0]
    assert event["status"] == "not_proposed"
    assert manager.event_audio(rid, manager._event(rid, event["id"]))[0][:4] == b"RIFF"
    assert (
        manager.event_audio(rid, manager._event(rid, rejected["items"][0]["id"]))[0][:4]
        == b"RIFF"
    )
    manager.review_event(rid, event["id"], "keep", 0)
    version = manager.reviewed_export(rid, 1)
    with zipfile.ZipFile(manager.reviewed_download(rid, version["id"])) as z:
        assert json.loads(z.read("selected-events.jsonl"))["id"] == event["id"]
    assert (
        manager.detail(rid, view="all", sample=True)["items"]
        == manager.detail(rid, view="all", sample=True)["items"]
    )
    assert manager.get(rid)["coverage"]["not_proposed_turns"] == 1
    with pytest.raises(KeyError):
        manager.review_event(rid, "../bad", "keep", 1)


def test_pause_is_bounded_and_resume_only_calls_remaining(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocking(body):
        calls.append(body)
        entered.set()
        release.wait(3)
        return backend(body)

    manager = Runs(tmp_path / "runs", backend=blocking)
    row = manager.submit(REQUIREMENT, upload=bundle(tmp_path, turns=many_turns()))
    rid = row["id"]
    assert entered.wait(2)
    manager.pause(rid)
    release.set()
    paused = finish(manager, rid)
    assert paused["status"] == "paused", paused
    assert 1 <= len(calls) <= 4 and paused["download_ready"]
    assert manager.detail(rid, view="unresolved")["total"] >= 4
    before = manager.download(rid)
    before_bytes = before.read_bytes()
    manager.resume(rid)
    assert finish(manager, rid)["status"] == "completed"
    assert len(calls) == 8
    assert before.read_bytes() == before_bytes and manager.download(rid) != before


def test_rate_limit_stops_scheduling_and_resume_retries_errors(tmp_path):
    calls = []

    def limited(body):
        calls.append(body)
        raise urllib.error.HTTPError(
            "https://provider.invalid", 429, "limited", {}, None
        )

    manager, rid = make_run(tmp_path, limited, turns=many_turns())
    paused = manager.get(rid)
    assert paused["status"] == "paused" and 1 <= len(calls) <= 4
    assert paused["unresolved"] == 8
    assert all(e["status"] != "complete" for e in manager.event_records(rid))
    manager.backend = backend
    manager.resume(rid)
    assert finish(manager, rid)["status"] == "completed"
    assert manager.get(rid)["errors"] == 0


def test_recovery_after_restart_retains_results_and_review(tmp_path):
    manager, rid = make_run(tmp_path)
    event = manager.detail(rid)["items"][0]
    manager.review_event(rid, event["id"], "keep", 0)
    manager.update(rid, status="curating")
    other = Runs(
        manager.root, backend=lambda _: pytest.fail("Must reuse completed result")
    )
    assert other.get(rid)["status"] == "interrupted"
    other.resume(rid)
    assert finish(other, rid)["status"] == "completed"
    assert other.reviews(rid)["items"][event["id"]]["decision"] == "keep"


def test_archive_rerun_delete_and_source_integrity(tmp_path):
    manager, rid = make_run(tmp_path)
    manager.archive(rid, True)
    assert manager.get(rid)["archived"]
    rerun = manager.rerun(rid)
    assert rerun["id"] != rid and rerun["status"] == "ready"
    assert manager.reviews(rerun["id"])["revision"] == 0
    manager.resume(rerun["id"])
    assert finish(manager, rerun["id"])["status"] == "completed"
    manager.delete(rid)
    assert len(manager.list()) == 1 and manager.cache.exists()
    event = manager.detail(rerun["id"])["items"][0]
    (manager.folder(rerun["id"]) / "assets/00000.wav").write_bytes(b"bad")
    with pytest.raises(Conflict):
        manager.event_audio(rerun["id"], event)


def test_new_api_routes_enforce_token_and_validate_reviews(tmp_path):
    app = create_app(tmp_path / "app", backend=backend)
    with TestClient(app) as client:
        example = client.get("/api/curation/example.zip")
        assert example.status_code == 200 and zipfile.is_zipfile(
            io.BytesIO(example.content)
        )
        token = client.get("/api/curation/bootstrap").json()["token"]
        client.headers["x-repair-token"] = token
        row = client.post(
            "/api/curation/runs/upload?preflight=true",
            content=bundle(tmp_path).read_bytes(),
        ).json()
        rid = row["id"]
        assert finish(app.state.runs, rid)["status"] == "ready"
        base = f"/api/curation/runs/{rid}"
        assert client.post(base + "/resume").status_code == 200
        assert finish(app.state.runs, rid)["status"] == "completed"
        event = client.get(base).json()["items"][0]
        review = base + f"/reviews/{event['id']}"
        assert (
            client.put(review, json={"decision": "bogus", "revision": 0}).status_code
            == 422
        )
        assert (
            client.put(review, json={"decision": "keep", "revision": 0}).status_code
            == 200
        )
        assert (
            client.put(review, json={"decision": "exclude", "revision": 0}).status_code
            == 409
        )
        assert client.get(base + f"/events/{event['id']}/audio").status_code == 200
        assert client.get(base + "/events/99999-0-9/audio").status_code == 404
        value = client.post(base + "/reviewed-exports", json={"revision": 1}).json()
        assert client.get(base + f"/reviewed-exports/{value['id']}").status_code == 200
        assert client.get(base + "?view=invalid").status_code == 422
        del client.headers["x-repair-token"]
        for method, path, body in [
            ("PUT", review, {"decision": "keep", "revision": 1}),
            ("POST", base + "/resume", {}),
            ("POST", base + "/pause", {}),
            ("POST", base + "/rerun", {}),
            ("PUT", base + "/archive", {"archived": True}),
            ("POST", base + "/reviewed-exports", {"revision": 1}),
            ("DELETE", base, {}),
        ]:
            assert client.request(method, path, json=body).status_code == 403


def test_retry_failed_items_does_not_rejudge_completed_results(tmp_path):
    calls = []

    def partial(body):
        packet = json.loads(body["contents"][0]["parts"][0]["text"])
        target = packet["target_turn_index"]
        calls.append(target)
        if target == 2:
            raise TimeoutError("no result")
        return backend(body)

    manager, rid = make_run(tmp_path, partial, turns=many_turns(3))
    assert manager.get(rid)["status"] == "completed_with_errors"
    assert sorted(calls) == [1, 2, 3]

    def recovered(body):
        calls.append(
            json.loads(body["contents"][0]["parts"][0]["text"])["target_turn_index"]
        )
        return backend(body)

    manager.backend = recovered
    manager.resume(rid)
    assert finish(manager, rid)["status"] == "completed"
    assert sorted(calls) == [1, 2, 2, 3]


def test_legacy_history_can_review_and_export_without_migration(tmp_path):
    import shutil

    def mixed(body):
        r = backend(body)
        target = json.loads(body["contents"][0]["parts"][0]["text"])[
            "target_turn_index"
        ]
        if target == 2:
            r["answer"].update(
                intent="backchannel", interruption_result="not_applicable"
            )
        return r

    manager, rid = make_run(tmp_path, mixed, turns=many_turns(2))
    folder = manager.folder(rid)
    shutil.copyfile(manager.download(rid), folder / "curated.zip")
    (folder / "prepared.json").unlink()
    (folder / "clips/00000-0-2.wav").unlink()
    manager.update(rid, workflow_version=1)
    assert manager.detail(rid)["legacy"]
    rejected = manager.detail(rid, view="not_selected")["items"][0]
    assert manager.event_audio(rid, rejected)[0][:4] == b"RIFF"
    manager.review_event(rid, rejected["id"], "keep", 0)
    version = manager.reviewed_export(rid, 1)
    with zipfile.ZipFile(manager.reviewed_download(rid, version["id"])) as z:
        assert json.loads(z.read("selected-events.jsonl"))["id"] == rejected["id"]
        assert "source-notices/LICENSE" in z.namelist()


def test_preflight_without_key_and_invalid_source_fail_before_model(tmp_path):
    manager = Runs(
        tmp_path / "runs",
        backend=lambda _: pytest.fail("No model calls"),
        configured=False,
    )
    row = manager.submit(REQUIREMENT, upload=bundle(tmp_path), preflight=True)
    assert finish(manager, row["id"])["status"] == "ready"
    with pytest.raises(ValueError, match="configured"):
        manager.resume(row["id"])
    bad = manager.submit(
        REQUIREMENT, upload=bundle(tmp_path, audio=False), preflight=True
    )
    assert finish(manager, bad["id"])["status"] == "failed"


def test_active_run_cannot_be_deleted_archived_or_review_exported(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def blocked(body):
        entered.set()
        release.wait(3)
        return backend(body)

    manager = Runs(tmp_path / "runs", backend=blocked)
    row = manager.submit(REQUIREMENT, upload=bundle(tmp_path))
    assert entered.wait(2)
    try:
        with pytest.raises(Conflict):
            manager.delete(row["id"])
        with pytest.raises(Conflict):
            manager.archive(row["id"], True)
        with pytest.raises(Conflict):
            manager.reviewed_export(row["id"], 0)
    finally:
        release.set()
    assert finish(manager, row["id"])["status"] == "completed"


def test_shutdown_drains_only_inflight_calls_and_preserves_resume(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocked(body):
        calls.append(body)
        entered.set()
        release.wait(3)
        return backend(body)

    manager = Runs(tmp_path / "runs", backend=blocked)
    row = manager.submit(REQUIREMENT, upload=bundle(tmp_path, turns=many_turns()))
    assert entered.wait(2)
    shutdown = threading.Thread(target=manager.shutdown)
    shutdown.start()
    release.set()
    shutdown.join(3)
    assert not shutdown.is_alive()
    assert 1 <= len(calls) <= 4
    assert manager.get(row["id"])["status"] == "paused"
    restored = Runs(manager.root, backend=backend)
    restored.resume(row["id"])
    assert finish(restored, row["id"])["status"] == "completed"
