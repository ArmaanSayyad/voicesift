"""Pinned audio-only curation evaluation on three independently labeled datasets."""

import argparse
import getpass
import hashlib
import io
import json
import os
import random
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
import soxr
from huggingface_hub import hf_hub_download

from repair_bench.curation.analysis import MODEL
from repair_bench.curation.audio_goals import parse_response, request_body
from repair_bench.curation.store import canonical

ROOT = Path("artifacts/voice-lanes")
LANES = {
    "intent": {
        "repo": "PolyAI/minds14",
        "revision": "40ce77cb32a384e4d50a568e1ec39ac804019d33",
        "files": ["en-US/train-00000-of-00001.parquet"],
        "folder": "PolyAI",
        "label": "intent_class",
        "per_class": None,
        "targets": {"balance": "balance", "freeze": "freeze"},
        "goals": {
            "balance": "The speaker's main request is to learn or check their bank account balance (how much money is available), not merely mention money, review individual transactions, or pay a bill.",
            "freeze": "The speaker's main request is to freeze, block, deactivate or stop use of a bank card, often because it is lost or stolen; also count asking what to do about a lost or stolen card. Do not count a card merely failing to work or being declined without a request to block it.",
        },
    },
    "vocal": {
        "repo": "DynamicSuperb/VocalSoundRecognition_VocalSound",
        "revision": "beb7fe456e01f1a9959daae8dd507fa7790f4b62",
        "files": ["data/test-00000-of-00001.parquet"],
        "folder": "DynamicSuperb",
        "label": "label",
        "per_class": 50,
        "targets": {"laughter": "Laughter", "cough": "Cough"},
        "goals": {
            "laughter": "The dominant vocal event is audible human laughter or chuckling. Do not count sighing, sniffing, sneezing, coughing, or throat clearing as laughter.",
            "cough": "The dominant vocal event is coughing: an explosive cough or series of coughs. Distinguish deliberate throat clearing, sneezing, sniffing, sighing and laughter. No inference about illness is requested.",
        },
    },
    "expression": {
        "repo": "xbgoose/ravdess",
        "revision": "a4a6c53ad083c4f16e92d1625e99113effe7569d",
        "files": [
            "data/train-00000-of-00002-94d632c9f1f51bbe.parquet",
            "data/train-00001-of-00002-bcaf733d4b46d6b2.parquet",
        ],
        "folder": "xbgoose",
        "label": "emotion",
        "per_class": 30,
        "targets": {"angry": "angry", "sad": "sad"},
        "goals": {
            "angry": "The dominant audible emotional expression of the spoken delivery is anger. Judge prosody and vocal delivery, not literal sentence meaning. Distinguish anger from disgust, fear, surprise, happiness, sadness, calm and neutral speech; loudness alone is insufficient. This describes performed expression, not the person's actual mental state.",
            "sad": "The dominant audible emotional expression of the spoken delivery is sadness. Judge prosody and vocal delivery, not literal sentence meaning. Distinguish sadness from calm, neutral, fearful, angry, disgusted, happy or surprised speech; quietness alone is insufficient. This describes performed expression, not the person's actual mental state.",
        },
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Frozen artifact changed: {path}"
    else:
        path.write_text(json.dumps(value, indent=2))


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "audio").mkdir(exist_ok=True)
    rng = random.Random(20260929)
    selection, sources = [], {}
    for lane, config in LANES.items():
        rows, files = [], []
        for name in config["files"]:
            path = Path(
                hf_hub_download(
                    config["repo"],
                    name,
                    repo_type="dataset",
                    revision=config["revision"],
                    local_dir=ROOT / "sources" / config["folder"],
                )
            )
            table = pq.read_table(path)
            part = table.to_pylist()
            if lane == "intent":
                names = json.loads(table.schema.metadata[b"huggingface"])["info"][
                    "features"
                ]["intent_class"]["names"]
                for r in part:
                    r["intent_class"] = names[r["intent_class"]]
            rows.extend(part)
            files.append(
                {
                    "file": name,
                    "bytes": path.stat().st_size,
                    "sha256": sha(path.read_bytes()),
                }
            )
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            groups[row[config["label"]]].append((index, row))
        chosen = []
        for label in sorted(groups):
            chosen.extend(
                rng.sample(groups[label], config["per_class"])
                if config["per_class"]
                else groups[label]
            )
        sources[lane] = {
            "repo": config["repo"],
            "revision": config["revision"],
            "files": files,
            "source_counts": {k: len(v) for k, v in groups.items()},
        }
        for index, row in chosen:
            raw = row["audio"]["bytes"]
            samples, sr = sf.read(io.BytesIO(raw), always_2d=True)
            assert samples.shape[1] <= 2 and 0 < len(samples) / sr <= 120
            samples = soxr.resample(samples, sr, 16000) if sr != 16000 else samples
            audio = io.BytesIO()
            sf.write(audio, samples, 16000, format="WAV", subtype="PCM_16")
            normalized = audio.getvalue()
            sid = sha(f"{lane}:{index}".encode())[:24]
            (ROOT / "audio" / f"{sid}.wav").write_bytes(normalized)
            selection.append(
                {
                    "id": sid,
                    "lane": lane,
                    "row_index": index,
                    "source_label": row[config["label"]],
                    "source_sha256": sha(raw),
                    "audio_sha256": sha(normalized),
                    "duration_s": len(samples) / 16000,
                    "speaker": row.get("actor"),
                    "truth": {
                        g: row[config["label"]] == target
                        for g, target in config["targets"].items()
                    },
                }
            )
    rng.shuffle(selection)
    assert len({r["source_sha256"] for r in selection}) == len(selection), (
        "Duplicate audio in evaluation"
    )
    protocol = {
        "model": MODEL,
        "seed": 20260929,
        "sources": sources,
        "requests": {
            lane: request_body(b"", config["goals"]) for lane, config in LANES.items()
        },
        "selection_sha256": sha(canonical(selection).encode()),
        "workers": 6,
        "scope": "Audio-only development evaluation of experimental curation goals. Labels and transcripts withheld; no prompt tuning or training. Two independent goals scored per audio request. Not production UI or full-dialogue evaluation.",
    }
    freeze(ROOT / "selection.json", selection)
    freeze(ROOT / "protocol.json", protocol)
    print(
        "Frozen",
        len(selection),
        "audio clips",
        dict(Counter(r["lane"] for r in selection)),
        flush=True,
    )
    return selection


def infer(row, key, response_folder="responses"):
    config = LANES[row["lane"]]
    audio = (ROOT / "audio" / f"{row['id']}.wav").read_bytes()
    assert sha(audio) == row["audio_sha256"]
    body = request_body(audio, config["goals"])
    digest = sha(canonical(body).encode())
    path = ROOT / response_folder / f"{row['id']}.json"
    if path.exists():
        cached = json.loads(path.read_text())
        assert cached["request_sha256"] == digest
        return cached
    result = {"id": row["id"], "lane": row["lane"], "request_sha256": digest}
    try:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            data=canonical(body).encode(),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.load(response)
        result.update(
            status="complete",
            answer=parse_response(data, config["goals"]),
            usage=data.get("usageMetadata", {}),
            model_version=data.get("modelVersion", MODEL),
        )
    except Exception as exc:  # noqa: BLE001 - record failures without leaking request credentials
        result.update(status="error", error=type(exc).__name__)
        if isinstance(getattr(exc, "code", None), int):
            result["http_status"] = exc.code
    path.write_text(json.dumps(result, indent=2))
    return result


def metrics(pairs):
    count = Counter()
    for truth, pred in pairs:
        count["positive" if truth else "negative"] += 1
        if pred in ["unclear", "error"]:
            count[pred] += 1
            count["missed_positive" if truth else "unresolved_negative"] += 1
        elif pred == "yes":
            count["tp" if truth else "fp"] += 1
        else:
            count["fn" if truth else "tn"] += 1
            if truth:
                count["missed_positive"] += 1
    n = len(pairs)
    ratio = lambda a, b: a / b if b else None
    return {
        "n": n,
        **{
            k: count[k]
            for k in [
                "positive",
                "negative",
                "tp",
                "fp",
                "tn",
                "fn",
                "missed_positive",
                "unclear",
                "error",
                "unresolved_negative",
            ]
        },
        "accuracy": ratio(count["tp"] + count["tn"], n),
        "precision": ratio(count["tp"], count["tp"] + count["fp"]),
        "recall": ratio(count["tp"], count["positive"]),
        "specificity": ratio(count["tn"], count["negative"]),
        "always_reject_accuracy": ratio(count["negative"], n),
    }


def summarize(selection, results):
    by_id = {r["id"]: r for r in results}
    assert len(by_id) == len(selection)
    goals = {}
    for lane, config in LANES.items():
        rows = [r for r in selection if r["lane"] == lane]
        for goal in config["goals"]:
            pairs = [
                (r["truth"][goal], by_id[r["id"]].get("answer", {}).get(goal, "error"))
                for r in rows
            ]
            goals[goal] = {"lane": lane, **metrics(pairs)}
    usage = Counter()
    for r in results:
        usage.update(
            {k: v for k, v in r.get("usage", {}).items() if isinstance(v, int)}
        )
    return {
        "audio_clips": len({r["audio_sha256"] for r in selection}),
        "requests": len(selection),
        "goals": goals,
        "usage": dict(usage),
        "errors": sum(r["status"] == "error" for r in results),
        "model_versions": dict(
            Counter(r.get("model_version", "error") for r in results)
        ),
    }


def export_selected(selection, results, root=None):
    """Create reviewable subsets; reference labels never become curated labels."""
    root = ROOT if root is None else root
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    by_id = {r["id"]: r for r in results}
    index = {}
    for lane, config in LANES.items():
        for goal, requirement in config["goals"].items():
            selected = [
                r
                for r in selection
                if r["lane"] == lane
                and by_id[r["id"]].get("answer", {}).get(goal) == "yes"
            ]
            path = exports / f"{goal}.zip"
            manifest = {
                "goal": goal,
                "requirement": requirement,
                "model": MODEL,
                "label_status": "model_selected_not_human_verified",
                "source": config["repo"],
                "revision": config["revision"],
                "selected": len(selected),
                "requested_clips": sum(r["lane"] == lane for r in selection),
                "failed_requests": sum(
                    r["lane"] == lane and by_id[r["id"]].get("status") == "error"
                    for r in selection
                ),
                "source_license_applies": True,
            }
            with zipfile.ZipFile(
                path.with_suffix(".tmp"), "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                records = []
                for row in selected:
                    audio = (root / "audio" / f"{row['id']}.wav").read_bytes()
                    assert sha(audio) == row["audio_sha256"]
                    name = f"audio/{row['id']}.wav"
                    archive.writestr(name, audio)
                    records.append(
                        {
                            "id": row["id"],
                            "audio": name,
                            "source_row_index": row["row_index"],
                            "audio_sha256": row["audio_sha256"],
                            "label_status": "model_selected_not_human_verified",
                        }
                    )
                archive.writestr(
                    "dataset.jsonl", "".join(canonical(r) + "\n" for r in records)
                )
                archive.writestr("manifest.json", json.dumps(manifest, indent=2))
                archive.writestr(
                    "README.txt",
                    "Experimental audio-only curated clips, not full conversations or human-verified labels.\n"
                    "Source licenses and attribution apply: MINDS-14 CC BY 4.0; VocalSound CC BY-SA 4.0; RAVDESS CC BY-NC-SA 4.0.\n"
                    "https://huggingface.co/datasets/PolyAI/minds14\n"
                    "https://github.com/YuanGongND/vocalsound\n"
                    "https://zenodo.org/records/1188976\n",
                )
            path.with_suffix(".tmp").replace(path)
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None
            index[goal] = {
                "selected": len(selected),
                "zip": str(path),
                "sha256": sha(path.read_bytes()),
            }
    (exports / "index.json").write_text(json.dumps(index, indent=2))
    return index


def main(argv=None, prepare_fn=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--retry-transport-errors",
        action="store_true",
        help="One separately recorded retry for every transport failure; never rejudge completed answers",
    )
    args = parser.parse_args(argv)
    selection = (prepare_fn or prepare)()
    if not args.run and not args.retry_transport_errors:
        return
    key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or getpass.getpass("Gemini key: ")
    )
    (ROOT / "responses").mkdir(exist_ok=True)
    results = []
    response_folder = "responses"
    pending = selection
    first_pass = None
    if args.retry_transport_errors:
        first_pass = [
            json.loads((ROOT / "responses" / f"{r['id']}.json").read_text())
            for r in selection
        ]
        failed_ids = {
            r["id"]
            for r in first_pass
            if r.get("error")
            in ["URLError", "TimeoutError", "ConnectionError", "HTTPError"]
        }
        pending = [r for r in selection if r["id"] in failed_ids]
        results = [r for r in first_pass if r["id"] not in failed_ids]
        response_folder = "transport-retry-1"
        (ROOT / response_folder).mkdir(exist_ok=True)
        freeze(
            ROOT / "transport-retry-protocol.json",
            {
                "max_retries": 1,
                "eligible_errors": [
                    "URLError",
                    "TimeoutError",
                    "ConnectionError",
                    "HTTPError",
                ],
                "ids": sorted(failed_ids),
                "rule": "Retry all transport failures once with the unchanged frozen request; completed predictions are never rerun.",
            },
        )
        (ROOT / "first-pass-summary.json").write_text(
            json.dumps(summarize(selection, first_pass), indent=2)
        )
        print("Transport retries", len(pending), flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [pool.submit(infer, row, key, response_folder) for row in pending]
        for job in as_completed(jobs):
            results.append(job.result())
            if len(results) % 50 == 0:
                print(
                    "Finished",
                    len(results),
                    "errors",
                    sum(r["status"] == "error" for r in results),
                    flush=True,
                )
    summary = summarize(selection, results)
    if first_pass is not None:
        summary["transport_retries"] = len(pending)
        summary["first_pass_errors"] = sum(r["status"] == "error" for r in first_pass)
    (ROOT / "resolved-results.json").write_text(
        json.dumps(sorted(results, key=lambda r: r["id"]), indent=2)
    )
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    export_selected(selection, results)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
