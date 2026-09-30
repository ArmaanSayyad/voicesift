"""Guard clean exports and conservative policy decisions, without API spend."""

import json

import pytest

from repair_bench.curation import precision
from repair_bench.curation.analysis import request_body
from repair_bench.curation.models import Review
from repair_bench.curation.store import Corpus, canonical


def answer(**kwargs):
    return dict(
        intent="take_floor",
        agent_outcome="stopped",
        interruption_result="successful",
        evidence_note="Incoming speaker gains the turn.",
        **kwargs,
    )


def packet():
    return {
        "target_turn_index": 1,
        "turns": [{"turn_index": 1}],
        "target_truncated": False,
        "context_truncated": False,
    }


@pytest.mark.parametrize(
    "intent,result,expected",
    [
        ("take_floor", "successful", "shortlist"),
        ("take_floor", "unsuccessful", "not_selected"),
        ("take_floor", "unknown", "needs_review"),
        ("normal_handoff", "not_applicable", "not_selected"),
        ("continuation", "not_applicable", "not_selected"),
        ("backchannel", "not_applicable", "not_selected"),
        ("unclear", "unknown", "needs_review"),
    ],
)
def test_shortlist_requires_success(intent, result, expected):
    a = answer()
    a.update(intent=intent, interruption_result=result)
    precision.validate(a)
    assert precision.disposition(a, packet()) == expected


@pytest.mark.parametrize(
    "change", [{"target_truncated": True}, {"context_truncated": True}, {"turns": []}]
)
def test_incomplete_evidence_never_shortlisted(change):
    assert precision.disposition(answer(), packet() | change) == "needs_review"


@pytest.mark.parametrize(
    "change",
    [
        {"intent": "backchannel"},
        {"interruption_result": "not_applicable"},
        {"evidence_note": ""},
        {"interruption_result": "made_up"},
    ],
)
def test_invalid_response_rejected(change):
    with pytest.raises(ValueError):
        precision.validate(answer() | change)


def test_legacy_requests_unchanged_and_production_has_no_channel_assumption():
    old = request_body({}, b"WAV")
    new = precision.request({}, b"WAV")
    assert old["contents"] == new["contents"]
    assert old["generationConfig"]["maxOutputTokens"] == 1024
    assert new["generationConfig"]["maxOutputTokens"] == 2048
    assert (
        "interruption_result"
        not in old["generationConfig"]["responseSchema"]["required"]
    )
    assert "left is assistant" not in precision.PROMPT


@pytest.mark.parametrize("result", ["unknown", "unsuccessful"])
def test_review_cannot_include_uncertain_or_failed_attempt(result):
    with pytest.raises(ValueError):
        Review(
            label="interruption",
            interruption_result=result,
            include=True,
            reviewer="test",
            expected_version=0,
        )


def test_legacy_included_review_requires_reconfirmation(tmp_path):
    c = Corpus(tmp_path)
    c.import_jsonl(
        json.dumps(
            {
                "id": "test",
                "source_group": "test",
                "provenance": "constructed fixture",
                "turns": [
                    {
                        "role": "assistant",
                        "text": "First",
                        "start_s": 0.0,
                        "end_s": 3.0,
                    },
                    {"role": "user", "text": "Wait", "start_s": 1.0, "end_s": 2.0},
                ],
            }
        )
    )
    cid = c.ids()[0]
    with c.db() as db:
        db.execute(
            "INSERT INTO reviews(candidate_id,version,body,created) VALUES (?,?,?,?)",
            (
                cid,
                1,
                canonical(
                    {
                        "label": "interruption",
                        "include": True,
                        "reviewer": "legacy",
                        "note": "",
                    }
                ),
                "test",
            ),
        )
    assert c.metrics()["included"] == 0
    with pytest.raises(ValueError):
        c.export()
    c.review(
        cid,
        Review(
            label="interruption",
            interruption_result="successful",
            include=True,
            reviewer="test",
            expected_version=1,
        ),
    )
    assert c.metrics()["included"] == 1
    export = c.export()
    row = json.loads((c.root / "exports" / f"{export['id']}.jsonl").read_text())
    assert row["schema_version"] == 3
    assert row["annotation"]["interruption_result"] == "successful"
