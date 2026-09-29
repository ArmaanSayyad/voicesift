"""Freeze speaker-separated few-shot examples and a uniform candidate sample."""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from report_turnbench import match_events

ROOT = Path("artifacts/turnbench")
OUT = Path("artifacts/prompt-ablation")
SEED = 20260930


def main():
    OUT.mkdir(exist_ok=True)
    rows = {}
    for f in sorted((ROOT / "source/data").glob("*.parquet")):
        cols = ["conversation_id", "metadata"] + [
            f"speaker_{s}_annotation_a" for s in (1, 2)
        ]
        for r in pq.read_table(f, columns=cols).to_pylist():
            rows[r["conversation_id"]] = r
    candidates = json.loads((ROOT / "candidates.json").read_text())
    gold = json.loads((ROOT / "gold.json").read_text())
    matched = match_events(candidates, gold)
    groups = {}
    for cid, row in rows.items():
        actors = tuple(sorted(row["metadata"][f"speaker_{s}_actor_id"] for s in (1, 2)))
        groups.setdefault(actors, []).append(cid)
    pairs = sorted(groups)
    rng = random.Random(SEED)
    rng.shuffle(pairs)
    example_actors = {a for pair in pairs[:3] for a in pair}
    example_convs = {
        cid
        for pair, cids in groups.items()
        if set(pair) & example_actors
        for cid in cids
    }
    eligible = [c for c in candidates if c["conversation_id"] not in example_convs]
    targets = rng.sample(sorted(eligible, key=lambda c: c["id"]), 400)
    pool = [
        c
        for c in candidates
        if c["conversation_id"] in example_convs and c["id"] in matched
    ]
    rng.shuffle(pool)

    def details(c):
        row = rows[c["conversation_id"]]
        onset = c["onset_s"]
        speaker = c["speaker"]
        own = min(
            row[f"speaker_{speaker}_annotation_a"],
            key=lambda e: abs(e["start_s"] - onset),
        )
        others = [
            e
            for e in row[f"speaker_{3 - speaker}_annotation_a"]
            if e["text"].strip() and e["start_s"] < onset < e["end_s"]
        ]
        return own, others

    def choose(label, predicate, answer, note, number=1):
        selected = []
        for c in pool:
            if any(e["id"] == c["id"] for e in examples + selected):
                continue
            g = gold[matched[c["id"]]]
            own, others = details(c)
            if (
                g["label"] == label
                and own["end_s"] - own["start_s"] <= 15
                and predicate(own, others, c)
            ):
                selected.append(
                    {
                        "id": c["id"],
                        "answer": answer,
                        "note": note,
                        "source_label": label,
                        "target_text": own["text"],
                        "opposing_text": [e["text"] for e in others],
                    }
                )
                if len(selected) == number:
                    return selected
        raise ValueError(f"Insufficient example category {answer}")

    examples = []
    examples += choose(
        "Interruption",
        lambda o, others, c: o["label"] == "Floor-taking Competitive Interruption",
        "take_floor",
        "The target enters an unfinished substantive turn and competitively takes the floor.",
    )
    examples += choose(
        "Interruption",
        lambda o, others, c: o["label"] == "Floor-taking Cooperative Interruption",
        "take_floor",
        "A cooperative contribution can still interrupt an unfinished turn; agreement alone does not make it a backchannel.",
    )
    examples += choose(
        "NonFloorTakingInterruption",
        lambda o, others, c: "Non-floor Taking" in o["label"],
        "take_floor",
        "The speaker attempts to enter or redirect the ongoing turn. The attempt counts even though it does not win the floor.",
    )
    examples += choose(
        "Backchannel",
        lambda o, others, c: "Backchannel" in o["label"],
        "backchannel",
        "Brief listener acknowledgement supporting the current speaker without claiming the turn.",
        2,
    )
    examples += choose(
        "Turn",
        lambda o, others, c: (
            o["label"] == "Normal Turn"
            and any(
                e["label"] == "Normal Turn"
                and e["end_s"] - e["start_s"] > 2
                and e["end_s"] - c["onset_s"] < 0.45
                for e in others
            )
        ),
        "normal_handoff",
        "The target begins at a reasonable completion point of the other speaker. Small timing overlap at an ordinary handoff is not enough to call an interruption.",
        2,
    )
    examples += choose(
        "Turn",
        lambda o, others, c: (
            o["label"] == "Strong Floor Hold"
            and any("Backchannel" in e["label"] for e in others)
        ),
        "continuation",
        "The target already holds the speaking turn and continues over a listener acknowledgement. The acknowledgement does not transfer floor ownership.",
    )
    # Raw labels/text are local only, never included in the scored target packet.
    manifest = {
        "seed": SEED,
        "example_conversations": sorted(example_convs),
        "evaluation_conversations": sorted({c["conversation_id"] for c in targets}),
        "population_candidates": len(eligible),
        "targets": [c["id"] for c in targets],
        "examples": examples,
        "design": "Uniform sample of 400 eligible candidates without using model predictions or ground-truth class. Example speakers excluded from targets. All already-used TurnBench dev data; no fresh-test claim.",
    }
    assert not example_convs & {c["conversation_id"] for c in targets}
    test_actors = {
        rows[c["conversation_id"]]["metadata"][f"speaker_{s}_actor_id"]
        for c in targets
        for s in (1, 2)
    }
    assert not example_actors & test_actors
    path = OUT / "selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest | {"targets": f"{len(targets)} IDs omitted"}, indent=2))


if __name__ == "__main__":
    main()
