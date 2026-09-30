"""Evaluate the production policy on the frozen development sample. No retry cherry-picking."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from report_turnbench import POS, match_events, metrics

from repair_bench.curation.analysis import MODEL, gemini
from repair_bench.curation.precision import VERSION, disposition, request
from repair_bench.curation.store import canonical

ROOT = Path("artifacts/prompt-ablation")
OUT = Path("artifacts/precision-evaluation")


def main():
    OUT.mkdir(exist_ok=True)
    selection = json.loads((ROOT / "selection.json").read_text())
    index = json.loads((ROOT / "input-index.json").read_text())
    protocol = {
        "policy": VERSION,
        "model": MODEL,
        "sample": selection["targets"],
        "scope": "Adaptive development evaluation on previously inspected sample, not held-out validation.",
        "request_template": request({}, b""),
        "inputs": index,
    }
    p = OUT / "protocol.json"
    if p.exists():
        assert json.loads(p.read_text()) == protocol, "Frozen protocol changed"
    else:
        p.write_text(json.dumps(protocol, indent=2))

    def run(cid):
        source = Path(index[cid]["directory"])
        packet = json.loads((source / "packet.json").read_text())
        audio = (source / "audio.wav").read_bytes()
        assert (
            hashlib.sha256(canonical(packet).encode()).hexdigest()
            == index[cid]["packet_sha256"]
        )
        assert hashlib.sha256(audio).hexdigest() == index[cid]["audio_sha256"]
        body = request(packet, audio)
        digest = hashlib.sha256((MODEL + canonical(body)).encode()).hexdigest()
        dest = OUT / (hashlib.sha256(cid.encode()).hexdigest()[:20] + ".json")
        if dest.exists():
            row = json.loads(dest.read_text())
            assert row["request_sha256"] == digest
            return row
        start = time.monotonic()
        try:
            result = gemini(body)
            row = {
                **result,
                "status": "complete",
                "disposition": disposition(result["answer"], packet),
            }
        except Exception as exc:  # noqa: BLE001 - preserve failed requests without logging data
            row = {
                "status": "error",
                "error": type(exc).__name__,
                "disposition": "needs_review",
            }
        row.update(id=cid, request_sha256=digest, seconds=time.monotonic() - start)
        dest.write_text(json.dumps(row, indent=2))
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for future in as_completed(
            [pool.submit(run, cid) for cid in selection["targets"]]
        ):
            rows.append(future.result())
            if len(rows) % 50 == 0:
                print(
                    "Completed",
                    len(rows),
                    "errors",
                    sum(r["status"] == "error" for r in rows),
                    flush=True,
                )
    rows.sort(key=lambda r: r["id"])
    gold = json.loads(Path("artifacts/turnbench/gold.json").read_text())
    candidates = json.loads(Path("artifacts/turnbench/candidates.json").read_text())
    match = match_events(candidates, gold)
    scored = [r for r in rows if r["id"] in match]
    u = {
        k: sum(r.get("usage", {}).get(k, 0) for r in rows)
        for k in ["promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount"]
    }
    summary = {
        "policy": VERSION,
        "total_events": len(rows),
        "scored_events": len(scored),
        "successful": metrics(
            [
                (gold[match[r["id"]]]["label"] in POS, r["disposition"] == "shortlist")
                for r in scored
            ]
        ),
        "shortlisted": sum(r["disposition"] == "shortlist" for r in rows),
        "unscored_shortlisted": sum(
            r["disposition"] == "shortlist" for r in rows if r["id"] not in match
        ),
        "errors": sum(r["status"] == "error" for r in rows),
        "usage": u,
        "estimated_recorded_usage_usd": (
            u["promptTokenCount"] * 0.75
            + (u["candidatesTokenCount"] + u["thoughtsTokenCount"]) * 3.75
        )
        / 1e6,
        "scope": protocol["scope"],
    }
    (OUT / "results.json").write_text(json.dumps(rows, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
