import base64
import importlib.util
import io
import json
import urllib.error
import zipfile
from pathlib import Path

import pytest

from repair_bench.curation.audio_goals import parse_response, request_body

spec = importlib.util.spec_from_file_location(
    "voice_eval", Path(__file__).parents[1] / "scripts/evaluate_voice_lanes.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_audio_goal_request_only_contains_audio_and_requirement():
    body = request_body(b"audio bytes", {"laugh": "Audible laughter"})
    parts = body["contents"][0]["parts"]
    assert len(parts) == 2
    assert json.loads(parts[0]["text"]) == {
        "curation_requirements": {"laugh": "Audible laughter"}
    }
    assert base64.b64decode(parts[1]["inlineData"]["data"]) == b"audio bytes"
    assert set(parts[1]["inlineData"]) == {"mimeType", "data"}


def test_incomplete_and_invalid_predictions_are_not_silent_negatives():
    def envelope(answer, finish="STOP"):
        return {
            "candidates": [
                {
                    "finishReason": finish,
                    "content": {"parts": [{"text": json.dumps(answer)}]},
                }
            ]
        }

    good = {"laugh": "yes", "evidence_note": "Laughter audible"}
    assert parse_response(envelope(good), {"laugh": "Audible laughter"}) == good
    for answer in [
        {"laugh": "maybe", "evidence_note": "?"},
        {"evidence_note": "missing"},
        dict(good, extra=True),
    ]:
        with pytest.raises(ValueError):
            parse_response(envelope(answer), {"laugh": "Audible laughter"})
    with pytest.raises(ValueError):
        parse_response(envelope(good, "MAX_TOKENS"), {"laugh": "Audible laughter"})


def test_metrics_include_failures_and_abstentions_in_accuracy_denominator():
    m = evaluation.metrics(
        [
            (True, "yes"),
            (True, "no"),
            (True, "error"),
            (False, "yes"),
            (False, "no"),
            (False, "unclear"),
        ]
    )
    assert m["accuracy"] == 2 / 6
    assert m["precision"] == 1 / 2 and m["recall"] == 1 / 3
    assert m["missed_positive"] == 2 and m["unresolved_negative"] == 1
    assert m["n"] == m["tp"] + m["fp"] + m["tn"] + m["fn"] + m["error"] + m["unclear"]
    assert evaluation.metrics([(False, "no")])["precision"] is None


def test_summary_distinguishes_repeated_requests_from_unique_audio(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "LANES",
        {
            "pair_one": {"goals": {"happy": "happy speech"}},
            "pair_two": {"goals": {"calm": "calm speech"}},
        },
    )
    selection = [
        {
            "id": "a",
            "lane": "pair_one",
            "audio_sha256": "same-recording",
            "truth": {"happy": True},
        },
        {
            "id": "b",
            "lane": "pair_two",
            "audio_sha256": "same-recording",
            "truth": {"calm": False},
        },
    ]
    results = [
        {"id": "a", "status": "complete", "answer": {"happy": "yes"}},
        {"id": "b", "status": "complete", "answer": {"calm": "no"}},
    ]
    summary = evaluation.summarize(selection, results)
    assert summary["audio_clips"] == 1 and summary["requests"] == 2
    assert summary["goals"]["happy"]["accuracy"] == 1
    assert summary["goals"]["calm"]["accuracy"] == 1


def test_exports_select_only_yes_and_withhold_reference_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    (tmp_path / "audio").mkdir()
    rows, results = [], []
    for index, decision in enumerate(["yes", "no", "unclear"]):
        sid = str(index)
        data = b"fixture audio"
        (tmp_path / "audio" / f"{sid}.wav").write_bytes(data)
        rows.append(
            {
                "id": sid,
                "lane": "vocal",
                "row_index": index,
                "audio_sha256": evaluation.sha(data),
                "source_label": "GOLD",
            }
        )
        results.append({"id": sid, "answer": {"laughter": decision, "cough": "no"}})
    exports = evaluation.export_selected(rows, results)
    assert exports["laughter"]["selected"] == 1
    assert exports["cough"]["selected"] == 0
    with zipfile.ZipFile(exports["laughter"]["zip"]) as archive:
        assert archive.testzip() is None
        records = archive.read("dataset.jsonl").decode()
        assert "GOLD" not in records
        assert json.loads(records)["id"] == "0"
        assert len([p for p in archive.namelist() if p.startswith("audio/")]) == 1


def test_transport_retry_preserves_original_and_caches_completed_answer(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    for folder in ["audio", "responses", "transport-retry-1"]:
        (tmp_path / folder).mkdir()
    audio = b"audio fixture"
    (tmp_path / "audio/test.wav").write_bytes(audio)
    row = {"id": "test", "lane": "vocal", "audio_sha256": evaluation.sha(audio)}
    calls = []

    def provider(request, timeout):
        calls.append(request.data)
        if len(calls) == 1:
            raise urllib.error.URLError("connection test")
        answer = {"laughter": "yes", "cough": "no", "evidence_note": "mock"}
        return io.BytesIO(
            json.dumps(
                {
                    "candidates": [
                        {
                            "finishReason": "STOP",
                            "content": {"parts": [{"text": json.dumps(answer)}]},
                        }
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(evaluation.urllib.request, "urlopen", provider)
    assert evaluation.infer(row, "test-key")["status"] == "error"
    recovered = evaluation.infer(row, "test-key", "transport-retry-1")
    assert recovered["status"] == "complete"
    assert evaluation.infer(row, "test-key", "transport-retry-1") == recovered
    assert len(calls) == 2 and calls[0] == calls[1]
    assert (
        json.loads((tmp_path / "responses/test.json").read_text())["status"] == "error"
    )
