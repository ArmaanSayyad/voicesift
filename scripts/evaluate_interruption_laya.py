"""Pinned SID-Bench transcript-side evaluation; labels never enter model requests."""

import os

os.environ["USE_TF"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import contextlib
import hashlib
import json
import random
import re
import time
from pathlib import Path

ROOT = Path("artifacts/interruption-eval")
REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
SEED = 20260929
QUESTION = "An agent is speaking and this is incoming user speech. Does the user want to take a substantive speaking turn, or are they only acknowledging/listening? Classify interruption intent, not mere voice activity. A substantive comment, question, objection or request takes the floor. A brief supportive acknowledgement without taking the floor does not."
CRITERIA = {
    "interrupt": "The user intends to take the floor or redirect the conversation with substantive speech.",
    "non_interrupt": "The user is only giving a backchannel, acknowledgement, filler, or non-directed speech, without taking the floor.",
}
SCHEMA = {"intent": {"type": "choice", "instructions": QUESTION, "criteria": CRITERIA}}


def clean(text):
    return " ".join(re.sub(r"<break>", " ", text or "", flags=re.IGNORECASE).split())


def load_rows():
    raw = [
        json.loads(x) for x in (ROOT / "en_test_lines.jsonl").read_text().splitlines()
    ]
    # These examples were inspected to understand the file format, so exclude them.
    inspected = {x["audio"] for x in raw[:2]} | {
        x["audio"] for x in [r for r in raw if r["total_nonbreak"]][:3]
    }
    rows = [
        {
            "id": hashlib.sha256(x["audio"].encode()).hexdigest()[:20],
            "source_audio": x["audio"],
            "text": clean(x["text_with_break"]),
            "positive": not x["total_nonbreak"],
        }
        for x in raw
        if x["audio"] not in inspected
    ]
    assert all("<break>" not in x["text"] for x in rows)
    random.Random(SEED).shuffle(rows)
    return rows


def score(rows, predictions):
    pairs = [(x, predictions[x["id"]]) for x in rows if x["id"] in predictions]
    tp = sum(x["positive"] and p for x, p in pairs)
    fp = sum(not x["positive"] and p for x, p in pairs)
    tn = sum(not x["positive"] and not p for x, p in pairs)
    fn = sum(x["positive"] and not p for x, p in pairs)
    return {
        "n": len(pairs),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "accuracy": (tp + tn) / len(pairs) if pairs else None,
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        if tp + fn and tn + fp
        else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--tag", default="")
    parser.add_argument("--subset-audio", action="store_true")
    args = parser.parse_args()
    rows = load_rows()
    if args.subset_audio:
        subset = {
            r["id"] for r in json.loads((ROOT / "audio-manifest.json").read_text())
        }
        rows = [r for r in rows if r["id"] in subset]
    if args.asr:
        trans = {
            x["id"]: x
            for x in map(json.loads, (ROOT / "asr.jsonl").read_text().splitlines())
        }
        rows = [{**r, "text": trans[r["id"]]["text"]} for r in rows if r["id"] in trans]
    suffix = ("asr" if args.asr else "reference") + args.tag
    protocol = {
        "seed": SEED,
        "laya_model": args.model,
        "laya_model_revision": args.revision,
        "schema": SCHEMA,
        "input": "Only cleaned user transcript; agent speaking is a fixed task assumption. No file names, gold labels, duration or break times.",
        "subset": "First two rows and first three negative rows excluded after format inspection.",
        "reference_sha256": hashlib.sha256(
            (ROOT / "en_test_lines.jsonl").read_bytes()
        ).hexdigest(),
        "n": len(rows),
        "timing_claim": "No raw-audio or actual overlap detection evaluated.",
    }
    protocol_path = ROOT / f"protocol-{suffix}.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError(
            "Existing protocol differs; use a new --tag to avoid mixing runs."
        )
    protocol_path.write_text(json.dumps(protocol, indent=2))
    import laya
    import torch

    torch.set_num_threads(4)
    start = time.perf_counter()
    with contextlib.redirect_stdout(__import__("sys").stderr):
        model = laya.load(args.model, device="cpu", revision=args.revision)
    print("loaded", time.perf_counter() - start, flush=True)
    output = ROOT / f"predictions-{suffix}.jsonl"
    done = (
        {json.loads(x)["id"] for x in output.read_text().splitlines()}
        if output.exists()
        else set()
    )
    with output.open("a") as f:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            t = time.perf_counter()
            out = model.predict({"user_speech": r["text"]}, SCHEMA)["answers"]["intent"]
            p = {
                "id": r["id"],
                "positive": r["positive"],
                "prediction": out["choice"] == "interrupt",
                "answer": out,
                "seconds": time.perf_counter() - t,
                "input_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
            }
            f.write(json.dumps(p) + "\n")
            f.flush()
            if i % 100 == 0:
                print(suffix, i, len(rows), flush=True)
    predictions = [json.loads(x) for x in output.read_text().splitlines()]
    lookup = {x["id"]: x["prediction"] for x in predictions}
    baselines = {
        "all_positive": {r["id"]: True for r in rows},
        "all_negative": {r["id"]: False for r in rows},
        "four_words": {
            r["id"]: len(re.findall(r"\b[\w']+\b", r["text"])) >= 4 for r in rows
        },
        "explicit_interruption_keywords": {
            r["id"]: bool(
                re.search(
                    r"\b(wait|stop|hold on|actually|excuse me|let me|but|no)\b",
                    r["text"],
                    re.IGNORECASE,
                )
            )
            for r in rows
        },
    }
    summary = {
        "protocol": protocol,
        "laya": score(rows, lookup),
        "baselines": {name: score(rows, p) for name, p in baselines.items()},
        "latency_s": {
            "mean": sum(x["seconds"] for x in predictions) / len(predictions),
            "min": min(x["seconds"] for x in predictions),
            "max": max(x["seconds"] for x in predictions),
        },
    }
    # Fixed low-prevalence mixture, selected without consulting any predictions.
    rng = random.Random(SEED + 1)
    pos = [r for r in rows if r["positive"]]
    neg = [r for r in rows if not r["positive"]]
    count = min(len(pos), len(neg) // 9)
    if count:
        mix = rng.sample(pos, count) + rng.sample(neg, count * 9)
        rng.shuffle(mix)
        summary["ten_percent_mixture"] = {
            "ids": [x["id"] for x in mix],
            "laya": score(mix, lookup),
            "baselines": {name: score(mix, p) for name, p in baselines.items()},
        }
    (ROOT / f"summary-{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("protocol", "ten_percent_mixture")
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.asr and not args.tag:
        diagnostic = random.Random(SEED + 2).sample(rows, 200)
        reverse = {
            "intent": {
                **SCHEMA["intent"],
                "criteria": dict(reversed(list(CRITERIA.items()))),
            }
        }
        changed = 0
        revlookup = {}
        with (ROOT / "order-diagnostic.jsonl").open("w") as f:
            for r in diagnostic:
                out = model.predict({"user_speech": r["text"]}, reverse)["answers"][
                    "intent"
                ]
                pred = out["choice"] == "interrupt"
                revlookup[r["id"]] = pred
                changed += pred != lookup[r["id"]]
                f.write(
                    json.dumps({"id": r["id"], "prediction": pred, "answer": out})
                    + "\n"
                )
        summary["option_order_diagnostic"] = {
            "n": len(diagnostic),
            "flipped": changed,
            "original": score(diagnostic, lookup),
            "reversed": score(diagnostic, revlookup),
            "note": "Diagnostic only; reversed order is not selected as an improved test configuration.",
        }
        (ROOT / f"summary-{suffix}.json").write_text(json.dumps(summary, indent=2))
        print("order diagnostic", changed, flush=True)


if __name__ == "__main__":
    main()
