"""Evaluate six further expression filters on the frozen 240-clip RAVDESS sample."""

import copy
import json
import shutil
from pathlib import Path

import evaluate_voice_lanes as engine

from repair_bench.curation.audio_goals import request_body
from repair_bench.curation.store import canonical

BASE = Path("artifacts/voice-lanes")
OUT = Path("artifacts/more-expressions")
PAIRS = {
    "positive_expression": ("happy", "surprised"),
    "aversive_expression": ("fearful", "disgust"),
    "low_arousal_expression": ("calm", "neutral"),
}
DESCRIPTIONS = {
    "happy": "happiness or joy; distinguish it from surprise, anger or merely lively neutral speech",
    "surprised": "surprise or astonishment; distinguish it from happiness, fear or merely high pitch",
    "fearful": "fear or apprehension; distinguish it from sadness, anger or surprise",
    "disgust": "disgust or revulsion; distinguish it from anger, sadness or fear",
    "calm": "calmness or soothing, relaxed delivery, distinguishable from ordinary neutral delivery; do not count sadness or mere quietness",
    "neutral": "neutral delivery without a marked emotional expression; distinguish ordinary neutral delivery from distinctly soothing/relaxed calmness and from sadness",
}


def configurations():
    source = engine.LANES["expression"]
    lanes = {}
    for lane, pair in PAIRS.items():
        config = copy.deepcopy(source)
        config["targets"] = {goal: goal for goal in pair}
        config["goals"] = {
            goal: f"The dominant audible emotional expression of the spoken delivery is {DESCRIPTIONS[goal]}. Judge prosody and vocal delivery, not literal sentence meaning. Distinguish between neutral, calm, happy, sad, angry, fearful, disgusted and surprised expression. This describes performed expression, not the person's actual mental state."
            for goal in pair
        }
        lanes[lane] = config
    return lanes


def prepare():
    base_selection = json.loads((BASE / "selection.json").read_text())
    base_protocol = json.loads((BASE / "protocol.json").read_text())
    assert (
        engine.sha(canonical(base_selection).encode())
        == base_protocol["selection_sha256"]
    )
    clips = [r for r in base_selection if r["lane"] == "expression"]
    assert len(clips) == 240 and len({r["audio_sha256"] for r in clips}) == 240
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    selection = []
    # Interleave requirement pairs so an outage cannot affect only a single pair.
    for row in clips:
        audio_path = BASE / "audio" / f"{row['id']}.wav"
        assert engine.sha(audio_path.read_bytes()) == row["audio_sha256"]
        for lane, pair in PAIRS.items():
            sid = engine.sha(f"{lane}:{row['id']}".encode())[:24]
            shutil.copyfile(audio_path, OUT / "audio" / f"{sid}.wav")
            selection.append(
                {
                    **row,
                    "id": sid,
                    "base_id": row["id"],
                    "lane": lane,
                    "truth": {goal: row["source_label"] == goal for goal in pair},
                }
            )
    protocol = {
        "model": engine.MODEL,
        "source": base_protocol["sources"]["expression"],
        "base_selection_sha256": base_protocol["selection_sha256"],
        "selection_sha256": engine.sha(canonical(selection).encode()),
        "unique_audio_clips": 240,
        "requests": len(selection),
        "workers": 6,
        "request_templates": {
            lane: request_body(b"", c["goals"]) for lane, c in engine.LANES.items()
        },
        "scope": "Six new binary requirements, three pairs per audio clip; same sample as prior anger/sadness evaluation. Not an untouched holdout; no tuning or examples from results. Audio only, no source labels/transcripts/filenames in model requests.",
    }
    engine.freeze(OUT / "protocol.json", protocol)
    engine.freeze(OUT / "selection.json", selection)
    print("Frozen 240 clips, 720 requests, six new expression goals", flush=True)
    return selection


if __name__ == "__main__":
    engine.LANES = configurations()
    engine.ROOT = OUT
    engine.main(prepare_fn=prepare)
