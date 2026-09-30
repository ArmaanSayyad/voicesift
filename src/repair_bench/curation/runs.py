"""Persistent source-to-ZIP jobs, separate from human-reviewed legacy exports."""

import hashlib
import json
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf

from ..store import atomic_json
from . import precision
from .analysis import MODEL, evidence, gemini
from .detect import detect
from .sources import (
    REQUIREMENT,
    download_dataset,
    file_hash,
    prepare_records,
    repo_id,
    unpack,
)
from .store import Conflict, canonical, now

ACTIVE = {"queued", "downloading", "preparing", "curating", "packaging"}


class EvidenceSource:
    def __init__(self, root, record, detection, reverse=False):
        self.root = root
        turns = record["turns"]
        if reverse:
            turns = [
                {**t, "role": "user" if t["role"] == "assistant" else "assistant"}
                for t in turns
            ]
        self.value = {
            "conversation": {**record, "turns": turns},
            "detection": detection,
            "turn_index": detection["turn_index"],
            "audio": {
                "name": record["file"],
                "duration": record["duration"],
                "hash": record["audio_sha256"],
            },
        }

    def get(self, _):
        return self.value


class Runs:
    def __init__(self, root, backend=None, configured=True, downloader=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = self.root / "cache"
        self.cache.mkdir(exist_ok=True)
        self.backend = backend or gemini
        self.configured = configured
        self.downloader = downloader or download_dataset
        self.lock = threading.Lock()
        self.prediction_locks = {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        for row in self.list():
            if row["status"] in ACTIVE:
                self.update(
                    row["id"],
                    status="interrupted",
                    message="Server restarted. Submit again to reuse cached model results.",
                )

    def list(self):
        return sorted(
            [json.loads(p.read_text()) for p in self.root.glob("*/status.json")],
            key=lambda r: r["created"],
            reverse=True,
        )

    def folder(self, rid):
        if (
            not isinstance(rid, str)
            or len(rid) != 32
            or any(c not in "0123456789abcdef" for c in rid)
        ):
            raise KeyError(rid)
        path = self.root / rid
        if not path.is_dir():
            raise KeyError(rid)
        return path

    def get(self, rid):
        return json.loads((self.folder(rid) / "status.json").read_text())

    def update(self, rid, **fields):
        row = self.get(rid)
        row.update(fields, updated=now())
        atomic_json(self.folder(rid) / "status.json", row)
        return row

    def submit(self, requirement, *, upload=None, filename=None, url=None):
        if requirement != REQUIREMENT:
            raise ValueError("Only the fixed interruption requirement is supported")
        if bool(upload) == bool(url):
            raise ValueError("Choose either an upload or a Hugging Face URL")
        if not self.configured:
            raise ValueError("Gemini is not configured on the server")
        if url:
            repo_id(url)
        with self.lock:
            if any(r["status"] in ACTIVE for r in self.list()):
                raise Conflict("A dataset is already running. Wait for it to finish.")
            rid = uuid.uuid4().hex
            folder = self.root / rid
            folder.mkdir()
            if upload:
                shutil.move(str(upload), folder / "upload.zip")
            row = {
                "id": rid,
                "created": now(),
                "updated": now(),
                "status": "queued",
                "requirement": requirement,
                "source": url or Path(filename or "dataset.zip").name,
                "message": "Queued",
                "conversations": 0,
                "candidates": 0,
                "processed": 0,
                "selected_conversations": 0,
                "selected_events": 0,
                "errors": 0,
                "policy_version": precision.VERSION,
                "download_ready": False,
            }
            atomic_json(folder / "status.json", row)
            self.pool.submit(self.run, rid, url)
        return row

    def predict(self, packet, audio):
        body = precision.request(packet, audio)
        key = hashlib.sha256((MODEL + canonical(body)).encode()).hexdigest()
        with self.lock:
            cache_lock = self.prediction_locks.setdefault(key, threading.Lock())
        with cache_lock:
            path = self.cache / f"{key}.json"
            cached = path.exists()
            result = json.loads(path.read_text()) if cached else self.backend(body)
            precision.validate(result["answer"])
            if not cached:
                atomic_json(path, result)
        return {
            **result,
            "request_sha256": key,
            "cached": cached,
            "disposition": precision.disposition(result["answer"], packet),
        }

    def run(self, rid, url):
        folder = self.folder(rid)
        progress = lambda **kw: self.update(rid, **kw)
        try:
            if url:
                progress(status="downloading", message="Resolving Hugging Face dataset")
                source, provenance, _ = self.downloader(
                    url, self.root / "downloads", progress
                )
                adapter = provenance["adapter"]
            else:
                progress(status="preparing", message="Reading uploaded ZIP")
                source = unpack(folder / "upload.zip", folder / "input")
                provenance = {
                    "kind": "upload",
                    "name": self.get(rid)["source"],
                    "sha256": file_hash(folder / "upload.zip"),
                    "adapter": "normalized-jsonl",
                }
                adapter = "normalized-jsonl"
            progress(status="preparing", message="Validating transcripts and audio")
            assets = folder / "assets"
            records = prepare_records(source, assets, adapter, progress)
            tasks = []
            untimed = 0
            for i, r in enumerate(records):
                r["duration"] = sf.info(assets / r["file"]).duration
                r["audio_sha256"] = file_hash(assets / r["file"])
                for reverse in [False, True] if r["both_directions"] else [False]:
                    turns = (
                        [
                            {
                                **t,
                                "role": "user"
                                if t["role"] == "assistant"
                                else "assistant",
                            }
                            for t in r["turns"]
                        ]
                        if reverse
                        else r["turns"]
                    )
                    detections, coverage = detect({"turns": turns})
                    untimed += coverage["untimed_user_turns"]
                    tasks.extend((i, d, reverse) for d in detections)
            if len(tasks) > 10000:
                raise ValueError(
                    "This run exceeds 10,000 overlap candidates. Submit a smaller dataset."
                )
            atomic_json(folder / "source.json", provenance)
            progress(
                status="curating",
                message="Finding interruptions",
                conversations=len(records),
                candidates=len(tasks),
            )
            results = []

            def analyze(task):
                i, d, reverse = task
                event_id = f"{i:05d}-{int(reverse)}-{d['turn_index']}"
                result = {
                    "id": event_id,
                    "conversation_index": i,
                    "source_conversation_id": records[i]["id"],
                    "target_turn_index": d["turn_index"],
                    "roles_reversed": reverse,
                    "detection": d,
                    "policy_version": precision.VERSION,
                    "model": MODEL,
                }
                try:
                    packet, audio, meta = evidence(
                        EvidenceSource(folder, records[i], d, reverse), event_id
                    )
                    prediction = self.predict(packet, audio)
                    result.update(prediction, evidence=meta, status="complete")
                    if result["disposition"] == "shortlist":
                        (folder / "clips" / f"{event_id}.wav").write_bytes(audio)
                except Exception as exc:  # noqa: BLE001 - retain a sanitized record for every failed event
                    result.update(
                        status="error",
                        disposition="needs_review",
                        error=type(exc).__name__,
                    )
                atomic_json(folder / "events" / f"{event_id}.json", result)
                return result

            (folder / "clips").mkdir()
            (folder / "events").mkdir()
            with ThreadPoolExecutor(max_workers=4) as pool:
                for f in as_completed([pool.submit(analyze, t) for t in tasks]):
                    results.append(f.result())
                    selected = [r for r in results if r["disposition"] == "shortlist"]
                    progress(
                        processed=len(results),
                        errors=sum(r["status"] == "error" for r in results),
                        selected_events=len(selected),
                        selected_conversations=len(
                            {r["conversation_index"] for r in selected}
                        ),
                    )
            results.sort(key=lambda r: r["id"])
            progress(status="packaging", message="Preparing ZIP download")
            selected = [r for r in results if r["disposition"] == "shortlist"]
            indices = sorted({r["conversation_index"] for r in selected})
            manifest = {
                **self.get(rid),
                "status": "completed_with_errors"
                if any(r["status"] == "error" for r in results)
                else "completed",
                "download_ready": True,
                "message": "Finished",
                "source_provenance": provenance,
                "label_status": "model_selected_not_human_verified",
                "selection": "At least one successful-interruption-v1 shortlist event; timing proposes candidates, Gemini judges audio and context.",
                "coverage": "All detected overlaps in supplied timed user turns; both directions for TurnBench human-human audio. No recall guarantee.",
                "audio": "Full selected conversations normalized to stereo/mono 16 kHz PCM16 WAV, plus event clips. Original channel order retained.",
                "untimed_turns": untimed,
                "event_errors": [r["id"] for r in results if r["status"] == "error"],
            }
            target = folder / "curated.zip"
            with zipfile.ZipFile(
                target.with_suffix(".tmp"), "w", compression=zipfile.ZIP_DEFLATED
            ) as z:
                rows = []
                for i in indices:
                    r = records[i]
                    row = {
                        k: v
                        for k, v in r.items()
                        if k not in ("file", "audio", "both_directions")
                    }
                    row.update(
                        audio=f"audio/{r['file']}",
                        curation={
                            "label_status": "model_selected_not_human_verified",
                            "events": [
                                e["id"]
                                for e in selected
                                if e["conversation_index"] == i
                            ],
                        },
                    )
                    rows.append(row)
                    z.write(assets / r["file"], row["audio"])
                z.writestr("dataset.jsonl", "".join(canonical(r) + "\n" for r in rows))
                z.writestr(
                    "selected-events.jsonl",
                    "".join(canonical(r) + "\n" for r in selected),
                )
                z.writestr(
                    "all-decisions.jsonl", "".join(canonical(r) + "\n" for r in results)
                )
                z.writestr("manifest.json", json.dumps(manifest, indent=2))
                z.writestr(
                    "README.txt",
                    "Machine-selected interruption candidates, not human-confirmed labels.\nReview before training. False positives and misses are expected.\nSee manifest.json for source, coverage and failures; dataset.jsonl contains full selected conversations.\nAn empty dataset.jsonl means no conversations were selected.\n",
                )
                for r in selected:
                    z.write(folder / "clips" / f"{r['id']}.wav", f"clips/{r['id']}.wav")
                for p in source.iterdir():
                    if p.is_file() and (
                        p.name.upper().startswith("LICENSE") or p.name == "README.md"
                    ):
                        z.write(p, f"source-notices/{p.name}")
            target.with_suffix(".tmp").replace(target)
            errors = sum(r["status"] == "error" for r in results)
            progress(
                status="completed_with_errors" if errors else "completed",
                message="Finished with analysis errors; see ZIP manifest"
                if errors
                else "Finished",
                download_ready=True,
                zip_sha256=file_hash(target),
            )
        except Exception as exc:  # noqa: BLE001 - never expose provider URLs, headers or credentials
            message = (
                str(exc)[:300]
                if type(exc) is ValueError
                else f"{type(exc).__name__}: could not process this dataset. Check the supported format and source access."
            )
            progress(status="failed", message=message, download_ready=False)

    def feedback(self, rid):
        path = self.folder(rid) / "feedback.json"
        return (
            json.loads(path.read_text())
            if path.exists()
            else {"text": "", "updated": None}
        )

    def save_feedback(self, rid, text):
        if not isinstance(text, str) or len(text) > 10000:
            raise ValueError("Feedback must be text of at most 10,000 characters")
        value = {"text": text, "updated": now()}
        # Feedback is separate from live progress and the immutable dataset export.
        with self.lock:
            atomic_json(self.folder(rid) / "feedback.json", value)
        return value

    def detail(self, rid, offset=0, limit=10):
        if offset < 0 or not 1 <= limit <= 25:
            raise ValueError("Use a nonnegative offset and a limit from 1 to 25")
        run = self.get(rid)
        result = {
            "run": run,
            "feedback": self.feedback(rid),
            "items": [],
            "offset": offset,
            "limit": limit,
            "total": 0,
        }
        if not run["download_ready"]:
            return result
        with zipfile.ZipFile(self.folder(rid) / "curated.zip") as archive:
            events = []
            with archive.open("selected-events.jsonl") as stream:
                for index, line in enumerate(stream):
                    result["total"] += 1
                    if offset <= index < offset + limit:
                        events.append(json.loads(line))
            wanted = {e["source_conversation_id"] for e in events}
            conversations = {}
            with archive.open("dataset.jsonl") as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["id"] in wanted:
                        conversations[row["id"]] = row
            for event in events:
                conversation = conversations[event["source_conversation_id"]]
                target = event["target_turn_index"]
                start = max(0, target - 10)
                turns = conversation["turns"][start : target + 11]
                result["items"].append(
                    {
                        "id": event["id"],
                        "conversation_id": conversation["id"],
                        "target_turn_index": target,
                        "roles_reversed": event["roles_reversed"],
                        "answer": event["answer"],
                        "evidence": event["evidence"],
                        "turns": [
                            {**t, "text": t["text"][:2000], "index": start + i}
                            for i, t in enumerate(turns)
                        ],
                        "context_truncated": start > 0
                        or target + 11 < len(conversation["turns"])
                        or any(len(t["text"]) > 2000 for t in turns),
                    }
                )
        return result

    def selected_clip(self, rid, event_id):
        # Only IDs in this run's exported selection are playable, never arbitrary paths.
        if (
            not isinstance(event_id, str)
            or not event_id
            or any(c not in "0123456789-" for c in event_id)
        ):
            raise KeyError(event_id)
        if not self.get(rid)["download_ready"]:
            raise Conflict("This run has no completed selection yet")
        with zipfile.ZipFile(self.folder(rid) / "curated.zip") as archive:
            with archive.open("selected-events.jsonl") as stream:
                event = next(
                    (r for line in stream if (r := json.loads(line))["id"] == event_id),
                    None,
                )
            if event is None:
                raise KeyError(event_id)
            audio = archive.read(f"clips/{event_id}.wav")
            if hashlib.sha256(audio).hexdigest() != event["evidence"]["clip_sha256"]:
                raise Conflict("Clip integrity check failed")
            return audio

    def download(self, rid):
        row = self.get(rid)
        if not row["download_ready"]:
            raise Conflict("This run has no completed ZIP yet")
        path = self.folder(rid) / "curated.zip"
        if file_hash(path) != row["zip_sha256"]:
            raise Conflict("Download integrity check failed")
        return path
