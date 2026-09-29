"""Transparent event matching and curation metrics; not TurnBench's online score."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path("artifacts/turnbench")
POS = {"Interruption"}
ATTEMPT = {"Interruption", "NonFloorTakingInterruption"}


def match_events(candidates, gold, tolerance=0.2):
    """One-to-one same-conversation/speaker onset matching, nearest pairs first."""
    edges = []
    groups = defaultdict(list)
    for j, g in enumerate(gold):
        groups[(g["conversation_id"], g["speaker"])].append((j, g))
    for c in candidates:
        for j, g in groups[(c["conversation_id"], c["speaker"])]:
            delta = abs(c["onset_s"] - g["start"])
            if delta <= tolerance + 1e-9:
                edges.append((delta, c["id"], j))
    matches = {}
    used = set()
    for _, cid, j in sorted(edges):
        if cid not in matches and j not in used:
            matches[cid] = j
            used.add(j)
    return matches


def metrics(pairs):
    counts = Counter(
        ("tp" if y else "fp") if pred else ("fn" if y else "tn") for y, pred in pairs
    )
    d = {k: counts[k] for k in ("tp", "fp", "tn", "fn")}
    d["precision"] = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else None
    d["recall"] = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else None
    return d


def main():
    candidates = json.loads((OUT / "candidates.json").read_text())
    gold = json.loads((OUT / "gold.json").read_text())
    conversations = json.loads((OUT / "conversations.json").read_text())
    results = (
        json.loads((OUT / "results.json").read_text())
        if (OUT / "results.json").exists()
        else []
    )
    lookup = {r["id"]: r for r in results}
    matches = match_events(candidates, gold)
    matched = {j: cid for cid, j in matches.items()}
    summary = {
        "conversations": len(conversations),
        "hours": sum(c["duration_s"] for c in conversations) / 3600,
        "candidates": len(candidates),
        "completed": len(results),
        "errors": sum(r["status"] == "error" for r in results),
        "unclear": sum(r["intent"] == "unclear" for r in results),
        "unscored_candidates": len(candidates) - len(matches),
        "model_flagged": sum(r["intent"] == "take_floor" for r in results),
        "flagged_unmatched": sum(
            r["intent"] == "take_floor" and r["id"] not in matches for r in results
        ),
        "gold_counts": dict(Counter(g["label"] for g in gold)),
        "timing_coverage": {
            label: {
                "total": sum(g["label"] == label for g in gold),
                "detected": sum(gold[j]["label"] == label for j in matched),
            }
            for label in sorted({g["label"] for g in gold})
        },
        "audio_status": dict(Counter(c["audio_status"] for c in conversations)),
    }
    if len(results) == len(candidates):
        cp = {c["id"]: c for c in candidates}
        predictors = {
            "gemini": lambda cid: lookup[cid]["intent"] == "take_floor",
            "four_words": lambda cid: cp[cid]["words"] >= 4,
            "timing_only": lambda cid: True,
            "review_queue": lambda cid: (
                lookup[cid]["intent"] in ("take_floor", "unclear")
                or lookup[cid]["status"] == "error"
            ),
        }
        for goal, positives in [
            ("successful", POS),
            ("successful_attempts_excluded", POS),
            ("attempt_inclusive", ATTEMPT),
        ]:
            eligible = [
                j
                for j, g in enumerate(gold)
                if goal != "successful_attempts_excluded"
                or g["label"] != "NonFloorTakingInterruption"
            ]
            summary[goal] = {}
            for name, pred in predictors.items():
                summary[goal][name] = {
                    "candidate_only": metrics(
                        [
                            (gold[j]["label"] in positives, pred(cid))
                            for cid, j in matches.items()
                            if j in eligible
                        ]
                    ),
                    "all_consensus_events": metrics(
                        [
                            (
                                gold[j]["label"] in positives,
                                pred(matched[j]) if j in matched else False,
                            )
                            for j in eligible
                        ]
                    ),
                }
            summary[goal]["conversations"] = metrics(
                [
                    (
                        any(
                            g["conversation_id"] == c["id"] and g["label"] in positives
                            for g in gold
                        ),
                        any(
                            r["intent"] == "take_floor"
                            for r in results
                            if cp[r["id"]]["conversation_id"] == c["id"]
                        ),
                    )
                    for c in conversations
                ]
            )
        summary["by_gold_label"] = {
            label: {
                "candidates": sum(gold[j]["label"] == label for j in matches.values()),
                "flagged": sum(
                    lookup[cid]["intent"] == "take_floor" and gold[j]["label"] == label
                    for cid, j in matches.items()
                ),
            }
            for label in sorted({g["label"] for g in gold})
        }
        summary["by_audio_status"] = {
            status: metrics(
                [
                    (gold[j]["label"] in ATTEMPT, lookup[cid]["intent"] == "take_floor")
                    for cid, j in matches.items()
                    if next(
                        c
                        for c in conversations
                        if c["id"] == gold[j]["conversation_id"]
                    )["audio_status"]
                    == status
                ]
            )
            for status in sorted({c["audio_status"] for c in conversations})
        }
        summary["target_truncated"] = sum(
            r["evidence"].get("target_truncated", False) for r in results
        )
        summary["context_truncated"] = sum(
            r["evidence"].get("context_truncated", False) for r in results
        )
        summary["model_versions"] = dict(
            Counter(r.get("model_version") for r in results)
        )
        summary["target_absent_from_text"] = sum(
            r["status"] == "complete"
            and r["evidence"]["target_turn_index"]
            not in [t["turn_index"] for t in r["evidence"]["turns"]]
            for r in results
        )
        summary["errors_by_gold_label"] = dict(
            Counter(
                gold[matches[r["id"]]]["label"] if r["id"] in matches else "Unmatched"
                for r in results
                if r["status"] == "error"
            )
        )
        summary["by_conversation_type"] = {}
        for kind in sorted({c["conversation_type"] for c in conversations}):
            ids = {c["id"] for c in conversations if c["conversation_type"] == kind}
            summary["by_conversation_type"][kind] = {
                "conversations": len(ids),
                "attempt_inclusive": metrics(
                    [
                        (
                            g["label"] in ATTEMPT,
                            lookup[matched[j]]["intent"] == "take_floor"
                            if j in matched
                            else False,
                        )
                        for j, g in enumerate(gold)
                        if g["conversation_id"] in ids
                    ]
                ),
            }
        summary["usage"] = {
            k: sum(r["usage"].get(k, 0) for r in results)
            for k in ["promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount"]
        }
        usage = summary["usage"]
        summary["estimated_successful_response_usd"] = (
            usage["promptTokenCount"] * 0.75
            + (usage["candidatesTokenCount"] + usage["thoughtsTokenCount"]) * 3.75
        ) / 1e6
        summary["cost_note"] = (
            "Published standard Gemini 3.8 Flash prices checked 2026-09-29: "
            "$0.75/M input, $3.75/M output including thinking. Failed/interrupted "
            "request usage is not available; this is not the complete billed cost. "
            "https://ai.google.dev/gemini-api/docs/pricing"
        )
        summary["matching_sensitivity"] = {}
        for tolerance in (0.1, 0.2, 0.3):
            alternative = match_events(candidates, gold, tolerance)
            inverse = {j: cid for cid, j in alternative.items()}
            summary["matching_sensitivity"][str(tolerance)] = {
                "unmatched_candidates": len(candidates) - len(alternative),
                "attempt_inclusive": metrics(
                    [
                        (
                            g["label"] in ATTEMPT,
                            lookup[inverse[j]]["intent"] == "take_floor"
                            if j in inverse
                            else False,
                        )
                        for j, g in enumerate(gold)
                    ]
                ),
            }
        # Resample whole conversations, not correlated snippets, for uncertainty.
        rng = np.random.default_rng(20260929)
        bootstrap_indices = rng.integers(
            0, len(conversations), (2000, len(conversations))
        )
        summary["conversation_bootstrap_95pct"] = {}
        for name in ("gemini", "four_words"):
            counts = []
            for conversation in conversations:
                m = metrics(
                    [
                        (
                            g["label"] in ATTEMPT,
                            predictors[name](matched[j]) if j in matched else False,
                        )
                        for j, g in enumerate(gold)
                        if g["conversation_id"] == conversation["id"]
                    ]
                )
                counts.append([m[k] for k in ("tp", "fp", "tn", "fn")])
            samples = np.asarray(counts)[bootstrap_indices].sum(axis=1)
            precision = samples[:, 0] / np.maximum(1, samples[:, 0] + samples[:, 1])
            recall = samples[:, 0] / np.maximum(1, samples[:, 0] + samples[:, 3])
            summary["conversation_bootstrap_95pct"][name] = {
                "precision": np.quantile(precision, [0.025, 0.975]).tolist(),
                "recall": np.quantile(recall, [0.025, 0.975]).tolist(),
                "resamples": 2000,
                "seed": 20260929,
            }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
