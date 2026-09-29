"""Timing-evidence diagnostic: same speech text, different occurrence of overlap."""

import os

os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hashlib
import json
import random
import time

import laya
import torch
from evaluate_interruption_laya import REVISION, ROOT, load_rows, score

from repair_bench.curation.detect import detect

schema = {
    "keep": {
        "type": "choice",
        "instructions": "Keep a conversation exactly when the user begins speaking strictly after an assistant speech interval starts and strictly before it ends. An equal endpoint is not an overlap. Use the numeric start_s and end_s values. Ignore whether the words sound like a correction or backchannel.",
        "criteria": {
            "keep": "At least one user speech onset falls strictly inside an assistant speech interval.",
            "reject": "No user speech onset falls strictly inside any assistant speech interval.",
        },
    }
}
samples = random.Random(20260933).sample(load_rows(), 50)
rows = []
for i, r in enumerate(samples):
    start = float((i % 5) + 1)
    end = start + 3
    for overlapping in (False, True):
        onset = start + 1.5 if overlapping else end + 0.25
        rows.append(
            {
                "id": r["id"] + str(overlapping),
                "positive": overlapping,
                "public": {
                    "turns": [
                        {
                            "role": "assistant",
                            "text": "Let me explain the available options.",
                            "start_s": start,
                            "end_s": end,
                        },
                        {
                            "role": "user",
                            "text": r["text"],
                            "start_s": onset,
                            "end_s": onset + 1,
                        },
                    ]
                },
            }
        )
random.Random(20260934).shuffle(rows)
(ROOT / "timing-protocol.json").write_text(
    json.dumps(
        {
            "schema": schema,
            "n": len(rows),
            "scope": "Constructed timing diagnostic; not naturally observed interactions",
            "model_revision": REVISION,
        },
        indent=2,
    )
)
torch.set_num_threads(4)
model = laya.load("convaiinnovations/laya", device="cpu", revision=REVISION)
predictions = {}
detector = {}
with (ROOT / "timing-predictions.jsonl").open("w") as f:
    for r in rows:
        t = time.perf_counter()
        answer = model.predict(r["public"], schema)["answers"]["keep"]
        predictions[r["id"]] = answer["choice"] == "keep"
        detector[r["id"]] = bool(detect(r["public"])[0])
        f.write(
            json.dumps(
                {
                    "id": r["id"],
                    "positive": r["positive"],
                    "prediction": predictions[r["id"]],
                    "answer": answer,
                    "seconds": time.perf_counter() - t,
                    "public_hash": hashlib.sha256(
                        json.dumps(r["public"], sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
            + "\n"
        )
        f.flush()
result = {
    "laya_with_timing": score(rows, predictions),
    "existing_detector": score(rows, detector),
    "scope": "50 transcript-matched pairs, constructed timing, no gold labels in classifier input.",
}
(ROOT / "timing-summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
