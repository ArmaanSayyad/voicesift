import io
import json
import threading
import time
import zipfile

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from repair_bench.curation.api import create_app
from repair_bench.curation.runs import Runs
from repair_bench.curation.sources import REQUIREMENT, repo_id, unpack
from repair_bench.curation.store import Conflict


def bundle(tmp_path, *, turns=None, audio=True):
    wav = io.BytesIO()
    sf.write(wav, np.zeros((80000, 2)), 16000, format="WAV")
    rows = [
        {
            "id": "test",
            "audio": "audio/source.wav",
            "turns": turns
            or [
                {
                    "role": "assistant",
                    "text": "One option is",
                    "start_s": 0.0,
                    "end_s": 3.0,
                },
                {"role": "user", "text": "Wait, please", "start_s": 1.0, "end_s": 4.0},
            ],
        }
    ]
    p = tmp_path / "upload.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("dataset.jsonl", json.dumps(rows[0]) + "\n")
        z.writestr("LICENSE", "Fixture license notice")
        if audio:
            z.writestr("audio/source.wav", wav.getvalue())
    return p


def backend(body):
    return {
        "answer": {
            "intent": "take_floor",
            "interruption_result": "successful",
            "agent_outcome": "stopped",
            "evidence_note": "Test double, not an acoustic judgment.",
        },
        "usage": {},
    }


def wait(runs, rid):
    for _ in range(400):
        r = runs.get(rid)
        if r["status"] in (
            "completed",
            "completed_with_errors",
            "failed",
            "interrupted",
        ):
            return r
        time.sleep(0.01)
    raise AssertionError("Job did not finish")


def test_upload_to_zip_and_history_cache(tmp_path):
    calls = []

    def count(body):
        calls.append(body)
        return backend(body)

    manager = Runs(tmp_path / "runs", backend=count)
    for _ in range(2):
        r = manager.submit(REQUIREMENT, upload=bundle(tmp_path), filename="calls.zip")
        done = wait(manager, r["id"])
        assert done["status"] == "completed", done
        assert done["selected_conversations"] == 1 and done["selected_events"] == 1
        with zipfile.ZipFile(manager.download(r["id"])) as z:
            assert "source-notices/LICENSE" in z.namelist()
            data = json.loads(z.read("dataset.jsonl"))
            assert data["audio"] in z.namelist()
            assert (
                data["curation"]["label_status"] == "model_selected_not_human_verified"
            )
            assert len(data["turns"]) == 2
            assert sf.info(io.BytesIO(z.read(data["audio"]))).duration == 5.0
            assert json.loads(z.read("manifest.json"))["status"] == "completed"
            assert len(z.read("all-decisions.jsonl").splitlines()) == 1
    assert len(calls) == 1 and len(manager.list()) == 2
    manager2 = Runs(tmp_path / "runs", backend=count)
    assert len(manager2.list()) == 2
    p = manager.download(r["id"])
    p.write_bytes(b"corrupt")
    with pytest.raises(Conflict):
        manager.download(r["id"])


def test_empty_selection_is_downloadable(tmp_path):
    def negative(body):
        r = backend(body)
        r["answer"].update(intent="backchannel", interruption_result="not_applicable")
        return r

    m = Runs(tmp_path / "runs", backend=negative)
    r = m.submit(REQUIREMENT, upload=bundle(tmp_path))
    done = wait(m, r["id"])
    assert done["status"] == "completed" and done["selected_conversations"] == 0
    with zipfile.ZipFile(m.download(r["id"])) as z:
        assert z.read("dataset.jsonl") == b""


def test_provider_failure_is_not_silently_negative(tmp_path):
    def error(body):
        raise RuntimeError("secret-data-must-not-leak")

    m = Runs(tmp_path / "runs", backend=error)
    r = m.submit(REQUIREMENT, upload=bundle(tmp_path))
    done = wait(m, r["id"])
    assert done["status"] == "completed_with_errors" and done["errors"] == 1
    with zipfile.ZipFile(m.download(r["id"])) as z:
        assert b"secret-data" not in z.read("all-decisions.jsonl")
        assert json.loads(z.read("manifest.json"))["event_errors"]


def test_reject_missing_audio_before_calls(tmp_path):
    def never(body):
        raise AssertionError("Model must not be called")

    m = Runs(tmp_path / "runs", backend=never)
    r = m.submit(REQUIREMENT, upload=bundle(tmp_path, audio=False))
    done = wait(m, r["id"])
    assert done["status"] == "failed" and not done["download_ready"]


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/datasets/a/b",
        "https://example.com/datasets/a/b",
        "https://huggingface.co.evil/datasets/a/b",
        "https://huggingface.co/datasets/a/b?x=1",
        "https://huggingface.co/datasets/a/b/tree/main",
        "https://user@huggingface.co/datasets/a/b",
    ],
)
def test_reject_ambiguous_or_external_links(url):
    with pytest.raises(ValueError):
        repo_id(url)


@pytest.mark.parametrize("name", ["../escape.wav", "/absolute.wav", "audio\\evil.wav"])
def test_zip_traversal(tmp_path, name):
    p = tmp_path / "bad.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(name, b"bad")
    with pytest.raises(ValueError):
        unpack(p, tmp_path / "dest")


def test_hf_job_uses_adapter_and_preserves_revision(tmp_path):
    source = unpack(bundle(tmp_path), tmp_path / "source")

    def downloader(url, cache, progress):
        assert repo_id(url) == "test/data"
        return (
            source,
            {
                "kind": "huggingface",
                "revision": "pinned-sha",
                "adapter": "normalized-jsonl",
            },
            [],
        )

    m = Runs(tmp_path / "runs", backend=backend, downloader=downloader)
    r = m.submit(REQUIREMENT, url="https://huggingface.co/datasets/test/data")
    assert wait(m, r["id"])["status"] == "completed"
    with zipfile.ZipFile(m.download(r["id"])) as z:
        assert (
            json.loads(z.read("manifest.json"))["source_provenance"]["revision"]
            == "pinned-sha"
        )


def test_fixed_requirement_busy_and_restart(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def block(body):
        entered.set()
        release.wait(3)
        return backend(body)

    m = Runs(tmp_path / "runs", backend=block)
    with pytest.raises(ValueError):
        m.submit("find laughter", upload=bundle(tmp_path))
    r = m.submit(REQUIREMENT, upload=bundle(tmp_path))
    assert entered.wait(2)
    with pytest.raises(Conflict):
        m.submit(REQUIREMENT, url="https://huggingface.co/datasets/a/b")
    release.set()
    assert wait(m, r["id"])["status"] == "completed"
    m.update(r["id"], status="curating")
    other = Runs(tmp_path / "runs", backend=backend)
    assert other.get(r["id"])["status"] == "interrupted"


def test_api_upload_download_and_requirement_enforcement(tmp_path):
    app = create_app(tmp_path / "app", backend=backend)
    with TestClient(app) as client:
        p = bundle(tmp_path)
        assert (
            client.post("/api/curation/runs/upload", content=p.read_bytes()).status_code
            == 403
        )
        client.headers["x-repair-token"] = client.get("/api/curation/bootstrap").json()[
            "token"
        ]
        assert (
            client.post(
                "/api/curation/runs/huggingface",
                json={
                    "url": "https://huggingface.co/datasets/a/b",
                    "requirement": "find laughter",
                },
            ).status_code
            == 422
        )
        response = client.post(
            "/api/curation/runs/upload?filename=test.zip", content=p.read_bytes()
        )
        assert response.status_code == 200, response.text
        r = response.json()
        assert wait(app.state.runs, r["id"])["status"] == "completed"
        assert len(client.get("/api/curation/runs").json()["items"]) == 1
        download = client.get(f"/api/curation/runs/{r['id']}/download")
        assert download.status_code == 200
        assert zipfile.is_zipfile(io.BytesIO(download.content))
        assert "dataset.jsonl" in client.get("/api/curation/format").text


def test_hf_download_pins_revision_and_only_referenced_files(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import huggingface_hub

    from repair_bench.curation.sources import download_dataset

    source = unpack(bundle(tmp_path), tmp_path / "source")
    files = [
        SimpleNamespace(rfilename=n, size=(source / n).stat().st_size)
        for n in ["dataset.jsonl", "audio/source.wav", "LICENSE"]
    ]
    files.append(SimpleNamespace(rfilename="unneeded.bin", size=10**12))
    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "dataset_info",
        lambda *a, **kw: SimpleNamespace(sha="immutable-sha", siblings=files),
    )
    calls = []

    def download(repo, name, **kwargs):
        calls.append((repo, name, kwargs))
        return str(source / name)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    _, meta, _ = download_dataset(
        "https://huggingface.co/datasets/test/data",
        tmp_path / "cache",
        lambda **kw: None,
    )
    assert meta["revision"] == "immutable-sha"
    assert all(c[2]["revision"] == "immutable-sha" for c in calls)
    assert "unneeded.bin" not in [c[1] for c in calls]


def test_turnbench_adapter_drops_gold_labels(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from repair_bench.curation.sources import prepare_records

    audio = io.BytesIO()
    sf.write(audio, np.zeros(80000), 16000, format="FLAC")
    row = {
        "conversation_id": "example",
        "speaker_1_audio": {"bytes": audio.getvalue()},
        "speaker_2_audio": {"bytes": audio.getvalue()},
        "speaker_1_annotation_a": [
            {"start_s": 0.0, "end_s": 3.0, "text": "A begins", "label": "SECRET_GOLD"}
        ],
        "speaker_2_annotation_a": [
            {"start_s": 1.0, "end_s": 4.0, "text": "B enters", "label": "SECRET_GOLD"}
        ],
    }
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), source / "data/part.parquet")
    result = prepare_records(
        source, tmp_path / "assets", "turnbench", lambda **kw: None
    )
    assert len(result) == 1 and result[0]["both_directions"]
    assert "SECRET_GOLD" not in json.dumps(result)
    assert sf.info(tmp_path / "assets" / result[0]["file"]).channels == 2


def test_oversized_upload_stream_is_rejected_and_cleaned(tmp_path, monkeypatch):
    from repair_bench.curation import api

    monkeypatch.setattr(api, "MAX_UPLOAD", 10)
    app = create_app(tmp_path / "app", backend=backend)
    with TestClient(app) as client:
        client.headers["x-repair-token"] = client.get("/api/curation/bootstrap").json()[
            "token"
        ]
        r = client.post("/api/curation/runs/upload", content=b"x" * 11)
        assert r.status_code == 413
        assert not list(app.state.runs.root.glob("*.zip"))


def test_no_overlap_is_successful_empty_run_without_model(tmp_path):
    def never(body):
        raise AssertionError("No model call for non-overlap")

    turns = [
        {"role": "assistant", "text": "Hello", "start_s": 0.0, "end_s": 1.0},
        {"role": "user", "text": "Hi", "start_s": 2.0, "end_s": 3.0},
    ]
    m = Runs(tmp_path / "runs", backend=never)
    r = m.submit(REQUIREMENT, upload=bundle(tmp_path, turns=turns))
    done = wait(m, r["id"])
    assert (
        done["status"] == "completed"
        and done["candidates"] == 0
        and done["errors"] == 0
    )
