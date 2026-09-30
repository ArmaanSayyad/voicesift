"""Persistent source-to-ZIP jobs, separate from human-reviewed legacy exports."""

import hashlib
import json
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import soundfile as sf

from ..store import atomic_json
from . import objectives, precision
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
from .workflow import Paused, Workflow

ACTIVE = {"queued", "downloading", "preparing", "curating", "packaging", "pausing"}


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


class Runs(Workflow):
    def __init__(self, root, backend=None, configured=True, downloader=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = self.root / "cache"
        self.cache.mkdir(exist_ok=True)
        self.backend = backend or gemini
        self.configured = configured
        self.downloader = downloader or download_dataset
        self.lock = threading.RLock()
        self.stops = {}
        self.prediction_locks = {}
        self.pool = ThreadPoolExecutor(max_workers=1)
        for row in self.list():
            if row["status"] in ACTIVE:
                self.update(
                    row["id"],
                    status="interrupted",
                    message="Server restarted. Resume to reuse completed results.",
                )

    def list(self):
        with self.lock:
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
        with self.lock:
            return json.loads((self.folder(rid) / "status.json").read_text())

    def update(self, rid, **fields):
        with self.lock:
            row = self.get(rid)
            row.update(fields, updated=now())
            atomic_json(self.folder(rid) / "status.json", row)
            return row

    def submit(
        self, requirement, *, upload=None, filename=None, url=None, preflight=False
    ):
        requirement = objectives.requirement(requirement)
        if bool(upload) == bool(url):
            raise ValueError("Choose either an upload or a Hugging Face URL")
        if not self.configured and not preflight:
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
                "source_url": url,
                "workflow_version": 2,
                "preflight": preflight,
                "archived": False,
            }
            atomic_json(folder / "status.json", row)
            self.stops[rid] = threading.Event()
            self.pool.submit(self.run, rid, url)
        return row

    def predict(self, packet, audio, objective=None, plan=None):
        body = (
            objectives.judge_request(audio, objective, plan)
            if objective
            else precision.request(packet, audio)
        )
        key = hashlib.sha256((MODEL + canonical(body)).encode()).hexdigest()
        with self.lock:
            cache_lock = self.prediction_locks.setdefault(key, threading.Lock())
        with cache_lock:
            path = self.cache / f"{key}.json"
            cached = path.exists()
            result = json.loads(path.read_text()) if cached else self.backend(body)
            (objectives.validate_answer if objective else precision.validate)(
                result["answer"]
            )
            if not cached:
                atomic_json(path, result)
        return {
            **result,
            "request_sha256": key,
            "cached": cached,
            "disposition": (
                {"yes": "shortlist", "no": "not_selected", "unclear": "needs_review"}[
                    result["answer"]["match"]
                ]
                if objective
                else precision.disposition(result["answer"], packet)
            ),
        }

    def run(self, rid, url):
        folder = self.folder(rid)
        stop = self.stops.setdefault(rid, threading.Event())
        objective = self.get(rid)["requirement"]

        def progress(**fields):
            if stop.is_set():
                raise Paused()
            return self.update(rid, **fields)

        try:
            snapshot = folder / "prepared.json"
            if snapshot.exists():
                prepared = json.loads(snapshot.read_text())
                if (
                    prepared["policy_version"]
                    != (
                        objectives.VERSION
                        if prepared.get("mode") == "audio"
                        else precision.VERSION
                    )
                    or prepared["model"] != MODEL
                ):
                    raise ValueError(
                        "This run used a different policy or model. Create a new run."
                    )
                records, tasks = prepared["records"], prepared["tasks"]
            else:
                if url:
                    progress(
                        status="downloading", message="Downloading and checking source"
                    )
                    source, provenance, _ = self.downloader(
                        url, self.root / "downloads", progress
                    )
                    adapter = provenance["adapter"]
                else:
                    progress(status="preparing", message="Checking uploaded ZIP")
                    # A cancelled extraction may leave a partial input directory.
                    shutil.rmtree(folder / "input", ignore_errors=True)
                    source = unpack(folder / "upload.zip", folder / "input")
                    provenance = {
                        "kind": "upload",
                        "name": self.get(rid)["source"],
                        "sha256": file_hash(folder / "upload.zip"),
                        "adapter": "normalized-jsonl",
                    }
                    adapter = "normalized-jsonl"
                progress(
                    status="preparing",
                    message="Validating transcripts and audio; no model calls",
                )
                records = prepare_records(source, folder / "assets", adapter, progress)
                mode = (
                    "interruption"
                    if objective == REQUIREMENT and all(r["turns"] for r in records)
                    else "audio"
                )
                policy = (
                    precision.VERSION if mode == "interruption" else objectives.VERSION
                )
                tasks, unscreened = [], []
                timed = untimed = 0
                for i, r in enumerate(records):
                    r["duration"] = sf.info(folder / "assets" / r["file"]).duration
                    r["audio_sha256"] = file_hash(folder / "assets" / r["file"])
                    if mode == "audio":
                        if r["duration"] > objectives.MAX_SECONDS:
                            raise ValueError(
                                "Freeform objectives require complete recordings of at most 5 minutes. Split longer recordings into meaningful examples before uploading; nothing is silently truncated."
                            )
                        if (
                            folder / "assets" / r["file"]
                        ).stat().st_size > objectives.MAX_AUDIO_BYTES:
                            raise ValueError(
                                "Freeform audio exceeds the 14 MB normalized WAV inline allowance. Use a shorter recording (about 3.6 minutes for stereo); nothing is truncated."
                            )
                        tasks.append(
                            (i, {"turn_index": 0, "whole_record": True}, False)
                        )
                        continue
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
                        timed += coverage["timed_user_turns"]
                        untimed += coverage["untimed_user_turns"]
                        tasks.extend((i, d, reverse) for d in detections)
                        covered = {
                            j for d in detections for j in d["user_turn_indices"]
                        }
                        unscreened.extend(
                            {
                                "id": f"{i:05d}-{int(reverse)}-{j}",
                                "conversation_index": i,
                                "source_conversation_id": r["id"],
                                "target_turn_index": j,
                                "roles_reversed": reverse,
                                "status": "not_proposed",
                                "disposition": "not_proposed",
                            }
                            for j, t in enumerate(turns)
                            if t["role"] == "user" and j not in covered
                        )
                if len(tasks) > 10000:
                    raise ValueError(
                        "This run exceeds 10,000 overlap candidates. Submit a smaller dataset."
                    )
                notices = folder / "source-notices"
                notices.mkdir(exist_ok=True)
                for p in source.iterdir():
                    if p.is_file() and (
                        p.name.upper().startswith("LICENSE") or p.name == "README.md"
                    ):
                        shutil.copyfile(p, notices / p.name)
                prepared = {
                    "records": records,
                    "tasks": tasks,
                    "unscreened": unscreened,
                    "provenance": provenance,
                    "policy_version": policy,
                    "mode": mode,
                    "requirement": objective,
                    "model": MODEL,
                    "coverage": {
                        "timed_turns": timed,
                        "untimed_turns": untimed,
                        "candidate_events": len(tasks),
                        "not_proposed_turns": len(unscreened),
                    },
                }
                atomic_json(snapshot, prepared)
                atomic_json(folder / "source.json", provenance)
            progress(
                conversations=len(records),
                candidates=len(tasks),
                coverage=prepared["coverage"],
                mode=prepared.get("mode", "interruption"),
                policy_version=prepared["policy_version"],
                compatibility="Complete audio validated"
                if prepared.get("mode") == "audio"
                else "Audio and timed speaker transcripts validated",
            )
            if self.get(rid).get("preflight"):
                progress(
                    status="ready",
                    message="Compatible. Ready for curation; no model calls made.",
                )
                return
            audio_mode = prepared.get("mode") == "audio"
            if audio_mode:
                plan_path = folder / "objective-plan.json"
                if plan_path.exists():
                    plan_result = json.loads(plan_path.read_text())
                else:
                    progress(status="curating", message="Interpreting your objective")
                    plan_body = objectives.plan_request(objective)
                    plan_result = self.backend(plan_body)
                    plan_result["request_sha256"] = hashlib.sha256(
                        (MODEL + canonical(plan_body)).encode()
                    ).hexdigest()
                    objectives.validate_plan(plan_result["answer"])
                    atomic_json(plan_path, plan_result)
                plan = plan_result["answer"]
                objectives.validate_plan(plan)
                self.update(rid, objective_plan=plan)
                if not plan["supported"]:
                    progress(status="needs_clarification", message=plan["reason"])
                    return
                if stop.is_set():
                    raise Paused()
            else:
                plan = None
            progress(
                status="curating",
                message="Evaluating complete recordings"
                if audio_mode
                else "Finding interruptions",
            )
            (folder / "clips").mkdir(exist_ok=True)
            (folder / "events").mkdir(exist_ok=True)
            results = []
            remaining = []
            for task in tasks:
                i, d, reverse = task
                eid = f"{i:05d}-{int(reverse)}-{d['turn_index']}"
                path = folder / "events" / f"{eid}.json"
                previous = json.loads(path.read_text()) if path.exists() else None
                if previous and previous["status"] == "complete":
                    results.append(previous)
                else:
                    remaining.append(task)

            def analyze(task):
                i, d, reverse = task
                eid = f"{i:05d}-{int(reverse)}-{d['turn_index']}"
                result = {
                    "id": eid,
                    "conversation_index": i,
                    "source_conversation_id": records[i]["id"],
                    "target_turn_index": d["turn_index"],
                    "roles_reversed": reverse,
                    "detection": d,
                    "policy_version": prepared["policy_version"],
                    "model": MODEL,
                }
                try:
                    if audio_mode:
                        audio, meta = self.event_audio(rid, result)
                        packet = None
                    else:
                        packet, audio, meta = evidence(
                            EvidenceSource(folder, records[i], d, reverse), eid
                        )
                    result.update(evidence=meta)
                    (folder / "clips" / f"{eid}.wav").write_bytes(audio)
                    result.update(
                        self.predict(
                            packet, audio, objective if audio_mode else None, plan
                        ),
                        status="complete",
                    )
                except Exception as exc:  # noqa: BLE001 - persist sanitized failure state without provider secrets
                    result.update(
                        status="error",
                        disposition="needs_review",
                        error=type(exc).__name__,
                    )
                    if getattr(exc, "code", None) in (401, 403, 429):
                        result["error"] = (
                            "RateLimit" if exc.code == 429 else "ProviderAccess"
                        )
                        stop.set()
                atomic_json(folder / "events" / f"{eid}.json", result)
                return result

            # Bounded scheduling: at most four calls in flight; pause/quota stops new calls.
            iterator = iter(remaining)
            with ThreadPoolExecutor(max_workers=4) as pool:
                pending = set()
                while True:
                    while len(pending) < 4 and not stop.is_set():
                        task = next(iterator, None)
                        if task is None:
                            break
                        pending.add(pool.submit(analyze, task))
                    if not pending:
                        break
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    results.extend(f.result() for f in done)
                    self.update(
                        rid,
                        processed=len(results),
                        errors=sum(r["status"] == "error" for r in results),
                    )
            # Retain every candidate, including errors not retried because the run paused.
            results = self.event_records(rid, include_unproposed=False)
            paused = stop.is_set()
            status = (
                "paused"
                if paused
                else (
                    "completed_with_errors"
                    if any(r["status"] != "complete" for r in results)
                    else "completed"
                )
            )
            self.update(
                rid,
                status="packaging",
                message="Preparing ZIP",
                processed=sum(r["status"] in ("complete", "error") for r in results),
            )
            self.package(rid, prepared, results, status)
        except Paused:
            self.update(rid, status="paused", message="Paused. Resume when ready.")
        except Exception as exc:  # noqa: BLE001 - persist sanitized failure state without provider secrets
            if getattr(exc, "code", None) in (401, 403, 429):
                self.update(
                    rid,
                    status="paused",
                    message=(
                        "Gemini rate limit reached while interpreting the objective. Wait before resuming."
                        if exc.code == 429
                        else "Gemini access failed while interpreting the objective. Check model access before resuming."
                    ),
                )
                return
            message = (
                str(exc)[:300]
                if type(exc) is ValueError
                else f"{type(exc).__name__}: could not process this dataset. Check the supported format and source access."
            )
            self.update(rid, status="failed", message=message)

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

    def legacy_clip(self, rid, event_id):
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
        path = self.folder(rid) / row.get("zip_path", "curated.zip")
        if file_hash(path) != row["zip_sha256"]:
            raise Conflict("Download integrity check failed")
        return path
