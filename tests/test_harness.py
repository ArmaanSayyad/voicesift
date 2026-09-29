import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from hypothesis import given, strategies as st

from repair_bench.core import (
    EventLog,
    Playback,
    Scheduler,
    TranscriptHistory,
    State,
    public_scenario,
    streaming_resample,
    overlap,
    yield_delay,
    RATE,
)
from repair_bench.cli import run_fixture


def test_deterministic_artifacts_and_replay(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ma, mb = run_fixture(a), run_fixture(b)
    assert ma == mb
    assert (a / "events.jsonl").read_bytes() == (b / "events.jsonl").read_bytes()
    assert len(EventLog.read(a / "events.jsonl").events) == ma["events"]
    audio, rate = sf.read(a / "rendered.wav")
    assert rate == RATE
    assert np.any(audio[:7200])
    assert not np.any(audio[7200:12000])
    assert np.any(audio[12000:16800])
    assert not ma["stale_output_rendered"]
    with pytest.raises(FileExistsError):
        run_fixture(a)


def test_corruption_and_clock_reversal(tmp_path):
    log = EventLog()
    log.emit(20, "x", {})
    with pytest.raises(ValueError):
        log.emit(19, "x", {})
    with pytest.raises(ValueError):
        log.emit(20, "x", {}, [1])
    with pytest.raises(ValueError):
        log.emit(20, "x", {"bad": float("nan")})
    path = tmp_path / "events.jsonl"
    log.save(path)
    path.write_text(path.read_text().replace('"kind":"x"', '"kind":"y"'))
    with pytest.raises(ValueError):
        EventLog.read(path)


def test_transcript_availability_not_audio_timestamp():
    h = TranscriptHistory()
    h.add("u", 0, "Friday", audio_end=100, available=180)
    h.add("u", 1, "Saturday", audio_end=160, available=240)
    assert h.visible(179) == {}
    assert h.visible(200)["u"]["text"] == "Friday"
    assert h.visible(240)["u"]["text"] == "Saturday"
    with pytest.raises(ValueError):
        h.add("u", 2, "bad", 300, 250)
    with pytest.raises(ValueError):
        h.add("u", 0, "old", 160, 250)


def test_public_context_does_not_leak_oracle():
    result = public_scenario(
        {
            "scenario_id": "x",
            "initial_utterance": "hello",
            "expected": {"answer": 42},
            "private_notes": "secret",
            "rubric": "test",
        }
    )
    assert set(result) == {"scenario_id", "initial_utterance"}


def test_state_preservation_and_stale_proposals():
    s = State({"day": "Friday", "time": "19:00"})
    s.patch({"day": "Saturday"}, 0, 0)
    assert s.values == {"day": "Saturday", "time": "19:00"}
    with pytest.raises(ValueError):
        s.patch({"day": "Monday"}, 0, 0)
    s.cancel()
    with pytest.raises(ValueError):
        s.patch({"day": "Tuesday"}, 1, 0)
    with pytest.raises(ValueError):
        s.patch({"unknown": "x"}, 1, 1)


@given(st.lists(st.integers(min_value=0, max_value=100), max_size=100))
def test_scheduler_stable_order(deadlines):
    s, seen = Scheduler(), []
    for i, deadline in enumerate(deadlines):
        s.at(deadline, lambda i=i: seen.append((s.now, i)))
    s.run()
    assert seen == sorted((t, i) for i, t in enumerate(deadlines))
    with pytest.raises(ValueError):
        s.at(s.now - 1, lambda: None)


@given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=30))
def test_stream_resampler_packetization(lengths):
    n = sum(lengths)
    signal = np.sin(np.arange(n, dtype=np.float32) * 0.02)
    pieces = np.split(signal, np.cumsum(lengths)[:-1])
    a = streaming_resample(pieces, 48000, 16000)
    b = streaming_resample([signal], 48000, 16000)
    assert len(a) == round(n / 3)
    np.testing.assert_allclose(a, b, atol=2e-6)


@pytest.mark.parametrize("rates", [(48000, 24000), (24000, 16000), (44100, 24000)])
def test_impulse_alignment_and_duration(rates):
    a, b = rates
    wave = np.zeros(a, dtype=np.float32)
    wave[a // 2] = 1
    out = streaming_resample(np.array_split(wave, 77), a, b)
    assert len(out) == b
    assert abs(np.argmax(out) - b // 2) <= 1


@given(
    st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=1000)
)
def test_cancel_partial_and_late_epochs(n, m):
    log = EventLog()
    p = Playback(log, capacity_samples=4000)
    assert p.enqueue(0, "a", 0, np.ones(n + m))
    assert np.all(p.render(0, n) == 1)
    p.cancel(n)
    assert not p.enqueue(n, "late", 0, np.ones(m))
    assert not np.any(p.render(n, m))
    assert p.enqueue(n + m, "b", 1, np.ones(5) * 0.25)
    np.testing.assert_array_equal(p.render(n + m, 5), np.ones(5) * 0.25)


def test_bounded_queue():
    p = Playback(EventLog(), capacity_samples=5)
    assert p.enqueue(0, "a", 0, np.ones(5))
    assert not p.enqueue(0, "b", 0, np.ones(1))
    assert p.queued == 5


def test_known_metrics_and_censoring():
    assert overlap([(0, 10), (5, 20)], [(8, 12)]) == 4
    assert yield_delay([(0, 10), (12, 16)], 5, 30, 4) == {
        "status": "observed",
        "samples": 11,
    }
    assert yield_delay([(0, 100)], 5, 30, 4) == {
        "status": "right_censored",
        "samples": 25,
    }
    assert yield_delay([], 5, 30, 4)["status"] == "not_applicable"
    assert yield_delay([(0, 10)], 5, 12, 4)["status"] == "right_censored"


def test_bundle_detects_audio_tampering_and_truncation(tmp_path):
    from repair_bench.cli import verify_bundle

    folder = tmp_path / "bundle"
    run_fixture(folder)
    assert verify_bundle(folder)["verified_artifacts"] == 4
    wav = folder / "rendered.wav"
    original = wav.read_bytes()
    wav.write_bytes(original[:-2])
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(folder)
    wav.write_bytes(original)
    events = folder / "events.jsonl"
    events.write_text("\n".join(events.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(folder)


def test_invalid_transcript_clocks():
    h = TranscriptHistory()
    for available in [float("nan"), 2.5, True, -1]:
        with pytest.raises(ValueError):
            h.add("u", 0, "x", 0, available)


def test_cancellation_at_arbitrary_sample_preserves_consumed_prefix():
    p = Playback(EventLog())
    p.enqueue(0, "speech", 0, np.ones(1000))
    consumed = p.render(0, 337)
    p.cancel(337)
    after = p.render(337, 663)
    assert np.all(consumed == 1) and np.all(after == 0)
    assert sum(x["samples"] for x in p.rendered) == 337
