"""Independent verification of the frozen first-pass shortlist, without prior answers."""

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_precision import OUT, ROOT
from report_turnbench import POS, match_events, metrics

from repair_bench.curation.analysis import MODEL, gemini
from repair_bench.curation.precision import disposition, verification_request
from repair_bench.curation.store import canonical


def main():
    rows = json.loads((OUT / "results.json").read_text())
    index = json.loads((ROOT / "input-index.json").read_text())
    dest = OUT / "verification"
    dest.mkdir(exist_ok=True)
    protocol = {
        "request_template": verification_request({}, b""),
        "first_pass_requests": {
            r["id"]: {
                "request_sha256": r["request_sha256"],
                "disposition": r["disposition"],
            }
            for r in sorted(rows, key=lambda r: r["id"])
        },
        "scope": "Adaptive second-stage development experiment; same model, independent prompt, no prior answer supplied.",
    }
    p = dest / "protocol.json"
    if p.exists():
        assert json.loads(p.read_text()) == protocol
    else:
        p.write_text(json.dumps(protocol, indent=2))

    def run(row):
        cid = row["id"]
        source = Path(index[cid]["directory"])
        packet = json.loads((source / "packet.json").read_text())
        audio = (source / "audio.wav").read_bytes()
        assert (
            hashlib.sha256(canonical(packet).encode()).hexdigest()
            == index[cid]["packet_sha256"]
        )
        assert hashlib.sha256(audio).hexdigest() == index[cid]["audio_sha256"]
        body = verification_request(packet, audio)
        fingerprint = hashlib.sha256((MODEL + canonical(body)).encode()).hexdigest()
        p = dest / (hashlib.sha256(cid.encode()).hexdigest()[:20] + ".json")
        if p.exists():
            result = json.loads(p.read_text())
            assert result["request_sha256"] == fingerprint
            return result
        start = time.monotonic()
        try:
            result = gemini(body)
            result.update(
                status="complete", disposition=disposition(result["answer"], packet)
            )
        except Exception as exc:  # noqa: BLE001 - persist sanitized failures
            result = {
                "status": "error",
                "error": type(exc).__name__,
                "disposition": "needs_review",
            }
        result.update(
            id=cid, request_sha256=fingerprint, seconds=time.monotonic() - start
        )
        p.write_text(json.dumps(result, indent=2))
        return result

    with ThreadPoolExecutor(max_workers=12) as pool:
        verified = list(
            pool.map(run, [r for r in rows if r["disposition"] == "shortlist"])
        )
    lookup = {r["id"]: r for r in verified}
    final = [
        dict(
            r,
            disposition=lookup[r["id"]]["disposition"]
            if r["id"] in lookup
            else r["disposition"],
        )
        for r in rows
    ]
    gold = json.loads(Path("artifacts/turnbench/gold.json").read_text())
    candidates = json.loads(Path("artifacts/turnbench/candidates.json").read_text())
    match = match_events(candidates, gold)
    summary = {
        "total_events": len(rows),
        "scored_events": sum(r["id"] in match for r in rows),
        "verification_requests": len(verified),
        "successful": metrics(
            [
                (gold[match[r["id"]]]["label"] in POS, r["disposition"] == "shortlist")
                for r in final
                if r["id"] in match
            ]
        ),
        "shortlisted": sum(r["disposition"] == "shortlist" for r in final),
        "unscored_shortlisted": sum(
            r["disposition"] == "shortlist" for r in final if r["id"] not in match
        ),
        "verification_errors": sum(r["status"] == "error" for r in verified),
        "usage": {
            k: sum(r.get("usage", {}).get(k, 0) for r in verified)
            for k in ["promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount"]
        },
    }
    (dest / "results.json").write_text(json.dumps(verified, indent=2))
    (dest / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
