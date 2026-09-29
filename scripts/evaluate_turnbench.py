"""Exercise the actual import/audio/analysis APIs on all prepared TurnBench candidates."""

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fastapi.testclient import TestClient

from repair_bench.curation.api import create_app

OUT = Path("artifacts/turnbench")
LOCK = threading.Lock()
DONE = 0


def run_window(w):
    global DONE
    wid = w["conversation"]["id"]
    app = create_app(OUT / "corpora" / wid)
    try:
        with TestClient(app) as c:
            c.headers["x-repair-token"] = c.get("/api/curation/bootstrap").json()[
                "token"
            ]
            r = c.post(
                "/api/curation/import", json={"jsonl": json.dumps(w["conversation"])}
            )
            assert r.status_code == 200, r.text
            rows = c.get("/api/curation/candidates").json()["items"]
            lookup = {x["turn_index"]: x for x in rows}
            targets = [(t, lookup[t["turn_index"]]) for t in w["targets"]]
            cid = targets[0][1]["conversation_id"]
            if not targets[0][1]["audio"]:
                r = c.post(
                    f"/api/curation/conversations/{cid}/audio",
                    content=Path(w["audio"]).read_bytes(),
                )
                assert r.status_code == 200, r.text
            pending = [x["id"] for _, x in targets if x["analysis"] is None]
            for offset in range(0, len(pending), 50):
                r = c.post(
                    "/api/curation/analyze",
                    json={"candidate_ids": pending[offset : offset + 50]},
                )
                assert r.status_code == 200, r.text
                job_id = r.json()["id"]
                deadline = time.monotonic() + 3600
                while True:
                    jobs = c.get("/api/curation/jobs").json()["items"]
                    job = next(j for j in jobs if j["id"] == job_id)
                    if job["status"] == "complete":
                        break
                    assert time.monotonic() < deadline, "Analysis deadline exceeded"
                    time.sleep(1)
            results = []
            for target, old in targets:
                row = app.state.corpus.get(old["id"])
                assert row["review"] is None
                a = row["analysis"]
                assert a is not None
                results.append(
                    {
                        "id": target["id"],
                        "window_id": wid,
                        "status": a["status"],
                        "intent": a.get("answer", {}).get("intent"),
                        "outcome": a.get("answer", {}).get("agent_outcome"),
                        "error": a.get("error"),
                        "usage": a.get("usage", {}),
                        "request_sha256": a.get("request_sha256"),
                        "evidence": a.get("evidence", {}),
                        "model_version": a.get("model_version"),
                        "seconds": a.get("seconds"),
                        "cached": a.get("cached"),
                    }
                )
            assert c.post("/api/curation/exports").status_code == 422
            (OUT / "results" / f"{wid}.json").write_text(json.dumps(results, indent=2))
            with LOCK:
                DONE += len(results)
                print(
                    "completed",
                    DONE,
                    "window",
                    wid,
                    "errors",
                    sum(x["status"] == "error" for x in results),
                    flush=True,
                )
            return results
    finally:
        app.state.analysis.pool.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-windows", type=int)
    args = p.parse_args()
    windows = json.loads((OUT / "windows.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert (
        hashlib.sha256(
            Path("src/repair_bench/curation/analysis.py").read_bytes()
        ).hexdigest()
        == protocol["production_analysis_sha256"]
    )
    # Freeze hashes before any model call, including all selected inputs and gold.
    frozen = {
        name: hashlib.sha256((OUT / f"{name}.json").read_bytes()).hexdigest()
        for name in ("protocol", "windows", "candidates", "gold", "conversations")
    }
    f = OUT / "frozen.json"
    if f.exists():
        assert json.loads(f.read_text()) == frozen, (
            "Inputs changed after evaluation started"
        )
    else:
        f.write_text(json.dumps(frozen, indent=2))
    (OUT / "results").mkdir(exist_ok=True)
    if args.limit_windows:
        windows = windows[: args.limit_windows]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_window, w) for w in windows]
        for future in as_completed(futures):
            results.extend(future.result())
    assert len({r["id"] for r in results}) == len(results)
    (OUT / ("pilot-results.json" if args.limit_windows else "results.json")).write_text(
        json.dumps(results, indent=2)
    )
    print(
        "TOTAL",
        len(results),
        "ERRORS",
        sum(r["status"] == "error" for r in results),
        flush=True,
    )


if __name__ == "__main__":
    main()
