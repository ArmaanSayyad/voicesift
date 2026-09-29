"""Full paired rerun of original/text-example prompts with a 2048-token cap."""

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from report_turnbench import ATTEMPT, POS, match_events, metrics
from run_prompt_ablation import MODEL, OUT, ROOT, call, load_input, make_body

from repair_bench.curation.store import canonical

DEST = OUT / "budget2048"
ARMS = ["original", "text_examples"]


def report(results):
    selection = json.loads((OUT / "selection.json").read_text())
    candidates = json.loads((ROOT / "candidates.json").read_text())
    gold = json.loads((ROOT / "gold.json").read_text())
    matched = match_events(candidates, gold)
    labels = {
        cid: gold[matched[cid]]["label"]
        for cid in selection["targets"]
        if cid in matched
    }
    summary = {
        "sample_candidates": 400,
        "scored_candidates": len(labels),
        "budget": 2048,
        "arms": {},
    }
    for arm in ARMS:
        rows = {r["id"]: r for r in results if r["arm"] == arm}
        assert len(rows) == 400
        u = {
            k: sum(r.get("usage", {}).get(k, 0) for r in rows.values())
            for k in ["promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount"]
        }
        summary["arms"][arm] = {
            name: metrics(
                [
                    (
                        label in pos,
                        rows[cid].get("answer", {}).get("intent") == "take_floor",
                    )
                    for cid, label in labels.items()
                ]
            )
            for name, pos in [("successful", POS), ("attempt_inclusive", ATTEMPT)]
        }
        summary["arms"][arm].update(
            errors=sum(r["status"] == "error" for r in rows.values()),
            incomplete=sum(
                r.get("error") == "incomplete_response" for r in rows.values()
            ),
            flags=sum(
                r.get("answer", {}).get("intent") == "take_floor" for r in rows.values()
            ),
            unscored_flags=sum(
                r.get("answer", {}).get("intent") == "take_floor"
                for cid, r in rows.items()
                if cid not in labels
            ),
            unclear=sum(
                r.get("answer", {}).get("intent") == "unclear" for r in rows.values()
            ),
            usage=u,
            estimated_recorded_usage_usd=(
                u["promptTokenCount"] * 0.75
                + (u["candidatesTokenCount"] + u["thoughtsTokenCount"]) * 3.75
            )
            / 1e6,
        )
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    DEST.mkdir(exist_ok=True)
    (DEST / "results").mkdir(exist_ok=True)
    selection = json.loads((OUT / "selection.json").read_text())
    index = json.loads((OUT / "input-index.json").read_text())
    protocol = {
        "parent_protocol_sha256": hashlib.sha256(
            (OUT / "protocol.json").read_bytes()
        ).hexdigest(),
        "arms": ARMS,
        "maxOutputTokens": 2048,
        "sample_candidates": 400,
        "requests": 800,
        "seed": 20260931,
        "reason": "Phase 1 text arm had 42 token-cap errors. Rerun ALL 400 items in original and text arms, not only failures. Only output cap changes; same inputs, prompts and examples. This is an adaptive development follow-up, not fresh validation.",
    }
    p = DEST / "protocol.json"
    if p.exists():
        assert json.loads(p.read_text()) == protocol
    else:
        p.write_text(json.dumps(protocol, indent=2))
    examples = [(e, load_input(e["id"], index)) for e in selection["examples"]]
    targets = {cid: load_input(cid, index) for cid in selection["targets"]}
    jobs = [(a, cid) for a in ARMS for cid in selection["targets"]]
    random.Random(20260931).shuffle(jobs)

    def run(job):
        arm, cid = job
        body = make_body(arm, targets[cid], examples)
        body["generationConfig"]["maxOutputTokens"] = 2048
        fingerprint = hashlib.sha256((MODEL + canonical(body)).encode()).hexdigest()
        path = (
            DEST
            / "results"
            / f"{arm}-{hashlib.sha256(cid.encode()).hexdigest()[:20]}.json"
        )
        if path.exists():
            result = json.loads(path.read_text())
            assert result["request_sha256"] == fingerprint
            return result
        start = time.monotonic()
        try:
            result = call(body)
        except Exception as exc:  # noqa: BLE001 - retain sanitized provider failures
            result = {"status": "error", "error": type(exc).__name__, "usage": {}}
        result.update(
            arm=arm,
            id=cid,
            request_sha256=fingerprint,
            seconds=time.monotonic() - start,
        )
        path.write_text(json.dumps(result, indent=2))
        return result

    results = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            results.append(f.result())
            if len(results) % 100 == 0:
                print("budget2048 completed", len(results), flush=True)
    assert len({(r["arm"], r["id"]) for r in results}) == 800
    (DEST / "results.json").write_text(json.dumps(results, indent=2))
    report(results)


if __name__ == "__main__":
    main()
