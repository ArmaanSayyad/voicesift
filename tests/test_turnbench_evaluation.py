"""Evaluation accounting must not hide missed events or double-count matches."""

import importlib.util
from pathlib import Path


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load("report_turnbench")
prepare = load("prepare_turnbench")


def test_matching_is_one_to_one_and_respects_speaker_and_conversation():
    gold = [{"conversation_id": "c", "speaker": 1, "start": 1.0}]
    candidates = [
        {"id": "far", "conversation_id": "c", "speaker": 1, "onset_s": 1.15},
        {"id": "near", "conversation_id": "c", "speaker": 1, "onset_s": 1.01},
        {"id": "speaker", "conversation_id": "c", "speaker": 2, "onset_s": 1.0},
        {"id": "conversation", "conversation_id": "d", "speaker": 1, "onset_s": 1.0},
    ]
    assert report.match_events(candidates, gold) == {"near": 0}


def test_matching_tolerance_boundary_and_unmatched():
    gold = [{"conversation_id": "c", "speaker": 1, "start": 1.0}]
    candidate = {"id": "x", "conversation_id": "c", "speaker": 1, "onset_s": 1.2}
    assert report.match_events([candidate], gold) == {"x": 0}
    candidate["onset_s"] = 1.201
    assert report.match_events([candidate], gold) == {}


def test_metrics_count_missed_positive_and_uncalled_negative():
    m = report.metrics([(True, True), (True, False), (False, True), (False, False)])
    assert m == {"tp": 1, "fp": 1, "tn": 1, "fn": 1, "precision": 0.5, "recall": 0.5}


def test_gold_labels_cannot_change_model_input():
    row = {
        f"speaker_{s}_annotation_a": [
            {"text": "hello", "start_s": 1.0, "end_s": 2.0, "label": "Normal Turn"},
            {"text": "", "start_s": 3.0, "end_s": 4.0, "label": "Channel Bleed"},
        ]
        for s in (1, 2)
    }
    before = prepare.turns_for(row, 1)
    for events in row.values():
        for e in events:
            e["label"] = "Floor-taking Competitive Interruption"
    assert prepare.turns_for(row, 1) == before
    assert len(before) == 2
    assert all("label" not in t for t in before)


def test_window_shifts_and_clips_without_time_stretching():
    turns = [
        {"role": "assistant", "text": "one", "start_s": 1.0, "end_s": 5.0},
        {"role": "user", "text": "two", "start_s": 4.0, "end_s": 7.0},
    ]
    window = prepare.window_turns(turns, 2.0, 6.0)
    assert [(t["start_s"], t["end_s"]) for t in window] == [(0.0, 3.0), (2.0, 4.0)]
    assert prepare.detect({"turns": window})[0][0]["user_onset_s"] + 2 == 4
