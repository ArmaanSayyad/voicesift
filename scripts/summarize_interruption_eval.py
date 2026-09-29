"""Summarize fixed predictions; never tune a classifier or threshold here."""

import json
import statistics
from pathlib import Path

from evaluate_interruption_laya import ROOT, load_rows, score


def calibration(preds):
    bins = []
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.000001)]:
        rows = [
            p for p in preds if lo <= max(p["answer"]["probabilities"].values()) < hi
        ]
        if rows:
            bins.append(
                {
                    "lo": lo,
                    "hi": min(hi, 1),
                    "n": len(rows),
                    "confidence": statistics.mean(
                        max(p["answer"]["probabilities"].values()) for p in rows
                    ),
                    "accuracy": statistics.mean(
                        p["prediction"] == p["positive"] for p in rows
                    ),
                }
            )
    return {
        "bins": bins,
        "ece": sum(b["n"] * abs(b["confidence"] - b["accuracy"]) for b in bins)
        / len(preds),
        "brier_positive": statistics.mean(
            (p["answer"]["probabilities"]["interrupt"] - int(p["positive"])) ** 2
            for p in preds
        ),
        "note": "Descriptive only, no calibration fitted; class proportions differ from deployment.",
    }


def main():
    ref = json.loads((ROOT / "summary-reference.json").read_text())
    asr = json.loads((ROOT / "summary-asr.json").read_text())
    refs = [
        json.loads(x)
        for x in (ROOT / "predictions-reference.jsonl").read_text().splitlines()
    ]
    preds = [
        json.loads(x) for x in (ROOT / "predictions-asr.jsonl").read_text().splitlines()
    ]
    ids = {p["id"] for p in preds}
    lookup = {p["id"]: p for p in refs}
    paired = [r for r in load_rows() if r["id"] in ids]
    result = {
        "reference": ref,
        "asr": asr,
        "paired_100": {
            "reference": score(paired, {p["id"]: p["prediction"] for p in refs}),
            "asr": asr["laya"],
            "changed_decision": sum(
                lookup[p["id"]]["prediction"] != p["prediction"] for p in preds
            ),
        },
        "calibration_reference": calibration(refs),
        "calibration_asr": calibration(preds),
    }
    # Same transcript cannot establish whether overlap occurred. Matched timing test
    # exercises the actual platform importer, not merely a reimplementation.
    import random
    import tempfile

    from repair_bench.curation.store import Corpus

    timing_rows = []
    for i, r in enumerate(paired):
        for overlap in (False, True):
            timing_rows.append(
                {
                    "id": r["id"] + ("-overlap" if overlap else "-handoff"),
                    "source_group": r["id"],
                    "split": "test",
                    "provenance": "Constructed timing counterfactual using a SID transcript. Agent utterance/timing invented; not a natural audio benchmark.",
                    "timing_source": "manual_annotations",
                    "turns": [
                        {
                            "role": "assistant",
                            "text": "I am explaining the available options.",
                            "start_s": 0.0,
                            "end_s": 4.0,
                        },
                        {
                            "role": "user",
                            "text": r["text"] or "(empty transcription)",
                            "start_s": 2.0 if overlap else 4.0,
                            "end_s": 5.0 if overlap else 7.0,
                        },
                    ],
                }
            )
    random.Random(20260929).shuffle(timing_rows)
    with tempfile.TemporaryDirectory() as folder:
        corpus = Corpus(Path(folder))
        for i in range(0, len(timing_rows), 100):
            corpus.import_jsonl(
                "\n".join(json.dumps(r) for r in timing_rows[i : i + 100])
            )
        found = {corpus.get(cid)["conversation"]["id"] for cid in corpus.ids()}
        unique = {}
        for r in timing_rows:
            key = tuple(
                (
                    t["role"],
                    " ".join(t["text"].lower().split()),
                    t["start_s"],
                    t["end_s"],
                )
                for t in r["turns"]
            )
            unique.setdefault(key, r)
        gold = {r["id"] for r in unique.values() if r["turns"][1]["start_s"] == 2.0}
        assert found == gold
        result["platform_timing_contract"] = {
            "input_conversations": len(timing_rows),
            "unique_conversations": len(unique),
            "duplicates_skipped": len(timing_rows) - len(unique),
            "overlap_positive": len(gold),
            "detected": len(found),
            "fp": len(found - gold),
            "fn": len(gold - found),
            "scope": "Constructed timestamps; verifies ingestion/detection, not real-world speaker/timestamp inference.",
            "text_only_identifiability": "Each transcript appears once with overlap and once without. A deterministic text-only classifier cannot distinguish the two; timing evidence is required.",
        }
    (ROOT / "report.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("reference", "asr")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
