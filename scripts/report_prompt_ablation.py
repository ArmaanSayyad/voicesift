"""Paired candidate metrics and conversation-cluster uncertainty for prompt arms."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from report_turnbench import ATTEMPT, POS, match_events, metrics
from run_prompt_ablation import ARMS

ROOT = Path("artifacts/turnbench")
OUT = Path("artifacts/prompt-ablation")


def main():
    selection = json.loads((OUT / "selection.json").read_text())
    candidates = json.loads((ROOT / "candidates.json").read_text())
    gold = json.loads((ROOT / "gold.json").read_text())
    # Match the entire candidate universe first, identically to the original report.
    matches = match_events(candidates, gold)
    lookup = {c["id"]: c for c in candidates}
    results = json.loads((OUT / "results.json").read_text())
    assert len(results) == 1600 and len({(r["arm"], r["id"]) for r in results}) == 1600
    by_arm = {a: {r["id"]: r for r in results if r["arm"] == a} for a in ARMS}
    sample = selection["targets"]
    scored = [cid for cid in sample if cid in matches]
    labels = {cid: gold[matches[cid]]["label"] for cid in scored}
    summary = {
        "sample_candidates": len(sample),
        "scored_candidates": len(scored),
        "unscored_candidates": len(sample) - len(scored),
        "evaluation_conversations": len(selection["evaluation_conversations"]),
        "example_conversations": len(selection["example_conversations"]),
        "counts": dict(Counter(labels.values())),
        "scope": "Candidate-only paired development comparison on 400 uniformly sampled existing timing candidates, 29 conversations. Not all conversations or end-to-end raw-audio accuracy.",
        "arms": {},
    }
    for arm, rows in by_arm.items():
        assert set(rows) == set(sample)
        flagged = lambda cid, rows=rows: (
            rows[cid].get("answer", {}).get("intent") == "take_floor"
        )
        usage = {
            key: sum(r.get("usage", {}).get(key, 0) for r in rows.values())
            for key in [
                "promptTokenCount",
                "candidatesTokenCount",
                "thoughtsTokenCount",
            ]
        }
        summary["arms"][arm] = {
            "successful": metrics(
                [(labels[cid] in POS, flagged(cid)) for cid in scored]
            ),
            "attempt_inclusive": metrics(
                [(labels[cid] in ATTEMPT, flagged(cid)) for cid in scored]
            ),
            "flags": sum(flagged(cid) for cid in sample),
            "unscored_flags": sum(flagged(cid) for cid in sample if cid not in labels),
            "errors": sum(r["status"] == "error" for r in rows.values()),
            "error_types": dict(
                Counter(r.get("error") for r in rows.values() if r["status"] == "error")
            ),
            "unclear": sum(
                r.get("answer", {}).get("intent") == "unclear" for r in rows.values()
            ),
            "intent_counts": dict(
                Counter(
                    r.get("answer", {}).get("intent") or "error" for r in rows.values()
                )
            ),
            "by_gold_label": {
                label: {
                    "total": sum(v == label for v in labels.values()),
                    "flagged": sum(
                        labels[cid] == label and flagged(cid) for cid in scored
                    ),
                }
                for label in sorted(set(labels.values()))
            },
            "usage": usage,
            "estimated_recorded_usage_usd": (
                usage["promptTokenCount"] * 0.75
                + (usage["candidatesTokenCount"] + usage["thoughtsTokenCount"]) * 3.75
            )
            / 1e6,
        }
    summary["four_words"] = {
        name: metrics(
            [(labels[cid] in positives, lookup[cid]["words"] >= 4) for cid in scored]
        )
        for name, positives in [("successful", POS), ("attempt_inclusive", ATTEMPT)]
    }
    old = {r["id"]: r for r in json.loads((ROOT / "results.json").read_text())}
    summary["original_repeatability"] = {
        "changed_binary_decisions": sum(
            (old[cid]["intent"] == "take_floor")
            != (by_arm["original"][cid].get("answer", {}).get("intent") == "take_floor")
            for cid in sample
        ),
        "previous_attempt_inclusive": metrics(
            [
                (labels[cid] in ATTEMPT, old[cid]["intent"] == "take_floor")
                for cid in scored
            ]
        ),
    }
    # Paired differences, resampling speaker pairs to preserve dependence across
    # multiple conversations recorded by the same two actors.
    import pyarrow.parquet as pq

    groups = {}
    for shard in sorted((ROOT / "source/data").glob("*.parquet")):
        for row in pq.read_table(
            shard, columns=["conversation_id", "metadata"]
        ).to_pylist():
            if row["conversation_id"] not in selection["evaluation_conversations"]:
                continue
            pair = tuple(
                sorted(row["metadata"][f"speaker_{s}_actor_id"] for s in (1, 2))
            )
            groups.setdefault(pair, []).append(row["conversation_id"])
    convs = list(groups.values())
    summary["bootstrap_cluster"] = {
        "unit": "speaker_pair",
        "count": len(convs),
        "resamples": 3000,
        "seed": 20260930,
    }
    rng = np.random.default_rng(20260930)
    indices = rng.integers(0, len(convs), (3000, len(convs)))
    distributions = {}
    for arm, rows in by_arm.items():
        per = []
        for conv in convs:
            m = metrics(
                [
                    (
                        labels[cid] in ATTEMPT,
                        rows[cid].get("answer", {}).get("intent") == "take_floor",
                    )
                    for cid in scored
                    if lookup[cid]["conversation_id"] in conv
                ]
            )
            per.append([m[k] for k in ["tp", "fp", "tn", "fn"]])
        x = np.array(per)[indices].sum(axis=1)
        distributions[arm] = {
            "precision": x[:, 0] / np.maximum(1, x[:, 0] + x[:, 1]),
            "recall": x[:, 0] / np.maximum(1, x[:, 0] + x[:, 3]),
        }
    summary["paired_bootstrap_differences_95pct"] = {
        arm: {
            m: np.quantile(
                distributions[arm][m] - distributions["original"][m], [0.025, 0.975]
            ).tolist()
            for m in ["precision", "recall"]
        }
        for arm in ARMS
        if arm != "original"
    }
    summary["total_estimated_recorded_usage_usd"] = sum(
        v["estimated_recorded_usage_usd"] for v in summary["arms"].values()
    )
    common_complete = [
        cid
        for cid in scored
        if all(by_arm[arm][cid]["status"] == "complete" for arm in ARMS)
    ]
    summary["common_complete_diagnostic"] = {
        "count": len(common_complete),
        "note": "Conditional diagnostic excluding every item with a failure in any arm; not a replacement for full-sample metrics.",
        "arms": {
            arm: metrics(
                [
                    (
                        labels[cid] in ATTEMPT,
                        rows[cid].get("answer", {}).get("intent") == "take_floor",
                    )
                    for cid in common_complete
                ]
            )
            for arm, rows in by_arm.items()
        },
    }
    summary["cost_note"] = (
        "Published standard rates $0.75/M input, $3.75/M output including thinking; includes incomplete responses with returned usage, excludes transport errors without usage. Not a billing statement."
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
