"""Pinned real-dialogue evaluation inputs; gold never enters model inputs."""

import argparse
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import soxr

from repair_bench.curation.detect import detect
from repair_bench.curation.models import Conversation

OUT = Path("artifacts/turnbench")
REV = "c29aa4e6422122a8dccbe23598016a089bea2121"
GOLD_REV = "76ccd045f121ccfa921abac2ad3107027e736911"


def turns_for(row, user):
    # Fixed annotator A supplies text/timing independently of consensus labels.
    # Empty transcripts cannot enter the current application's turn schema.
    return sorted(
        [
            {
                "role": "user" if s == user else "assistant",
                "text": e["text"],
                "start_s": float(e["start_s"]),
                "end_s": float(e["end_s"]),
            }
            for s in (1, 2)
            for e in row[f"speaker_{s}_annotation_a"]
            if e["text"].strip() and e["end_s"] > e["start_s"]
        ],
        key=lambda t: (t["start_s"], t["end_s"], t["role"]),
    )


def event_key(cid, speaker, onset):
    return f"{cid}:{speaker}:{onset:.4f}"


def window_turns(turns, start, stop):
    return [
        {
            **t,
            "start_s": max(t["start_s"], start) - start,
            "end_s": min(t["end_s"], stop) - start,
        }
        for t in turns
        if t["end_s"] > start and t["start_s"] < stop
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    actual = subprocess.check_output(
        ["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == GOLD_REV
    sys.path.insert(0, str(args.reference.resolve()))
    from turnbench.gold import consensus_for_conversation

    from repair_bench.curation.analysis import MODEL, PROMPT, SCHEMA

    (OUT / "windows").mkdir(exist_ok=True)
    gold_rows, windows, candidates, conversations = [], [], [], []
    for path in sorted((OUT / "source/data").glob("*.parquet")):
        columns = ["conversation_id", "metadata", "audio_status"] + [
            f"speaker_{s}_{suffix}"
            for s in (1, 2)
            for suffix in ["audio", "annotation_a", "annotation_b", "annotation_c"]
        ]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1, columns=columns):
            row = batch.to_pylist()[0]
            cid = row["conversation_id"]
            annotations = {
                (s, a): [
                    (e["start_s"], e["end_s"], e["label"], e["text"])
                    for e in row[f"speaker_{s}_annotation_{a}"]
                ]
                for s in (1, 2)
                for a in "abc"
            }
            gold, excluded = consensus_for_conversation(
                SimpleNamespace(annotations=annotations)
            )
            gold_rows.extend(dict(conversation_id=cid, **asdict(e)) for e in gold)
            audio = []
            source_hashes = []
            for s in (1, 2):
                raw = row[f"speaker_{s}_audio"]["bytes"]
                source_hashes.append(hashlib.sha256(raw).hexdigest())
                signal, rate = sf.read(io.BytesIO(raw), dtype="float32")
                assert signal.ndim == 1
                audio.append(soxr.resample(signal, rate, 16000))
            assert abs(len(audio[0]) - len(audio[1])) <= 1
            stereo = np.stack([a[: min(map(len, audio))] for a in audio], axis=1)
            duration = len(stereo) / 16000
            conversations.append(
                {
                    "id": cid,
                    "duration_s": duration,
                    "conversation_type": row["metadata"]["conversation_type"],
                    "audio_status": row["audio_status"],
                    "audio_sha256": source_hashes,
                    "excluded": [asdict(e) for e in excluded],
                    "empty_text_a": sum(
                        not e["text"].strip()
                        for s in (1, 2)
                        for e in row[f"speaker_{s}_annotation_a"]
                    ),
                }
            )
            for user in (1, 2):
                turns = turns_for(row, user)
                detections, _ = detect({"turns": turns})
                by_core = {}
                for d in detections:
                    key = event_key(cid, user, d["user_onset_s"])
                    target = turns[d["turn_index"]]
                    candidates.append(
                        {
                            "id": key,
                            "conversation_id": cid,
                            "speaker": user,
                            "onset_s": d["user_onset_s"],
                            "end_s": target["end_s"],
                            "words": len(
                                " ".join(
                                    turns[i]["text"] for i in d["user_turn_indices"]
                                ).split()
                            ),
                        }
                    )
                    by_core.setdefault(int(d["user_onset_s"] // 60), []).append(d)
                for core, targets in by_core.items():
                    # Full original candidate groups fit within context unless very long.
                    # Include source intervals out to their endpoints to preserve detector merging.
                    lo = max(0.0, core * 60 - 30.0)
                    hi = min(duration, (core + 1) * 60 + 60.0)
                    local = window_turns(turns, lo, hi)
                    wid = f"{cid}-s{user}-w{core}"
                    body = {
                        "id": wid,
                        "source_group": cid,
                        "split": "dev",
                        "provenance": "TurnBench dev natural human-human dialogue; human speakers mapped to user/assistant for compatibility. Original timing, no synthetic overlaps.",
                        "timing_source": "manual_annotations",
                        "turns": local,
                    }
                    Conversation.model_validate(body)
                    ds = detect(body)[0]
                    selected = []
                    for d in targets:
                        matched = [
                            x
                            for x in ds
                            if abs(x["user_onset_s"] + lo - d["user_onset_s"]) < 1e-6
                        ]
                        assert len(matched) == 1, (wid, d)
                        # Ensure target segmentation and evidence horizon survive windowing.
                        original_end = max(
                            turns[i]["end_s"] for i in d["user_turn_indices"]
                        )
                        needed = min(
                            duration,
                            original_end + 3,
                            max(0, d["user_onset_s"] - 3) + 30,
                        )
                        assert hi >= needed
                        selected.append(
                            {
                                "id": event_key(cid, user, d["user_onset_s"]),
                                "turn_index": matched[0]["turn_index"],
                            }
                        )
                    wav = OUT / "windows" / f"{wid}.wav"
                    # Agent channel first, incoming user second, consistent with existing fixtures.
                    samples = stereo[int(lo * 16000) : int(hi * 16000)]
                    if user == 1:
                        samples = samples[:, ::-1]
                    sf.write(wav, samples, 16000, subtype="PCM_16")
                    windows.append(
                        {
                            "conversation": body,
                            "audio": str(wav),
                            "source_offset_s": lo,
                            "targets": selected,
                        }
                    )
            print(
                "prepared",
                len(conversations),
                "conversations",
                len(candidates),
                "candidates",
                flush=True,
            )
    assert len(conversations) == 38
    assert len({c["id"] for c in candidates}) == len(candidates)
    protocol = {
        "dataset": "mundo-ai/turn-benchmark-dev",
        "dataset_revision": REV,
        "reference_revision": GOLD_REV,
        "model": MODEL,
        "prompt": PROMPT,
        "schema": SCHEMA,
        "generation": {"temperature": 0, "maxOutputTokens": 1024},
        "production_analysis_sha256": hashlib.sha256(
            Path("src/repair_bench/curation/analysis.py").read_bytes()
        ).hexdigest(),
        "input": "Annotator A nonempty text and original intervals; both role orientations; 60s core windows with 30s left and 60s right context; labels withheld.",
        "gold": "Official canonical 2-of-3 consensus, 200ms endpoint agreement. Match same-speaker candidate to nearest unmatched consensus start within 200ms, deterministic distance ordering. Report successful interruptions and attempts separately. No consensus match is unscored, not negative.",
        "evaluation": "Offline retrospective curation, not official online TurnBench latency scoring. No prompt tuning. All detected candidates evaluated once. Empty transcript annotations omitted by current application schema.",
        "conversation_count": len(conversations),
        "candidate_count": len(candidates),
        "window_count": len(windows),
        "gold_counts": dict(Counter(g["label"] for g in gold_rows)),
    }
    for name, data in [
        ("protocol", protocol),
        ("gold", gold_rows),
        ("conversations", conversations),
        ("candidates", candidates),
        ("windows", windows),
    ]:
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=2))
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
