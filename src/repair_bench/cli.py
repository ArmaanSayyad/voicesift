import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess

import numpy as np
import psutil
import soundfile as sf

from .core import EventLog, Playback, Scheduler, RATE, canonical


def run_fixture(path: Path):
    path.mkdir(parents=True, exist_ok=False)
    log, clock = EventLog(), Scheduler()
    player = Playback(log)
    length = RATE * 2
    output = np.zeros(length, dtype=np.float32)
    generated = (
        np.sin(2 * np.pi * 440 * np.arange(RATE) / RATE).astype(np.float32) * 0.2
    )
    input_pcm = np.zeros(length, dtype=np.float32)
    input_pcm[7200:12000] = 0.2 * np.sin(2 * np.pi * 220 * np.arange(4800) / RATE)
    clock.at(
        0,
        lambda: log.emit(
            0, "run.started", {"mode": "deterministic_fixture", "real_speech": False}
        ),
    )
    clock.at(0, lambda: player.enqueue(0, "original", 0, generated))
    clock.at(
        7200, lambda: log.emit(7200, "intervention.fired", {"requested_sample": 7200})
    )
    clock.at(7200, lambda: player.cancel(7200))
    clock.at(7680, lambda: player.enqueue(7680, "late-old", 0, generated[:480]))
    clock.at(12000, lambda: player.enqueue(12000, "repair", 1, generated[:4800]))
    for start in range(0, length, 480):

        def render(start=start):
            output[start : start + 480] = player.render(start, 480)

        clock.at(start, render)
    clock.run()
    log.emit(length, "run.completed", {"mode": "deterministic_fixture"})
    log.save(path / "events.jsonl")
    for name, audio in (
        ("input", input_pcm),
        ("generated", generated),
        ("rendered", output),
    ):
        sf.write(path / f"{name}.wav", audio, RATE, subtype="PCM_16")
    summary = {
        "fixture": True,
        "samples": length,
        "sample_rate": RATE,
        "cancellation_sample": 7200,
        "repair_sample": 12000,
        "stale_output_rendered": any(x["id"] == "late-old" for x in player.rendered),
        "events": len(log.events),
        "last_event_hash": log.events[-1]["hash"],
    }
    summary["artifacts"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir())
    }
    (path / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def verify_bundle(path: Path):
    """Check all expected artifact bytes and bind the event chain to the manifest."""
    manifest = json.loads((path / "manifest.json").read_text())
    expected = {"events.jsonl", "input.wav", "generated.wav", "rendered.wav"}
    if set(manifest["artifacts"]) != expected:
        raise ValueError("unexpected artifact set")
    for name, digest in manifest["artifacts"].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
    log = EventLog.read(path / "events.jsonl")
    if (
        len(log.events) != manifest["events"]
        or not log.events
        or log.events[-1]["hash"] != manifest["last_event_hash"]
    ):
        raise ValueError("event manifest mismatch")
    return {"verified_events": len(log.events), "verified_artifacts": len(expected)}


def doctor():
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "memory_bytes": psutil.virtual_memory().total,
        "free_disk_bytes": shutil.disk_usage(".").free,
        "ollama_available": shutil.which("ollama") is not None,
        "microphone_tested": False,
        "browser_playback_tested": False,
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("fixture")
    p.add_argument("output", type=Path)
    p = sub.add_parser("verify")
    p.add_argument("events", type=Path)
    p = sub.add_parser("verify-bundle")
    p.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if args.command == "doctor":
        result = doctor()
    elif args.command == "fixture":
        result = run_fixture(args.output)
    elif args.command == "verify-bundle":
        result = verify_bundle(args.bundle)
    else:
        result = {"verified_events": len(EventLog.read(args.events).events)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
