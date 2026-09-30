"""Review, recovery and immutable export versions for single-user dataset runs."""

import hashlib
import io
import json
import shutil
import threading
import uuid
import zipfile

import soundfile as sf

from ..store import atomic_json
from .sources import file_hash
from .store import Conflict, canonical, now


class Paused(Exception):
    pass


class Workflow:
    def shutdown(self):
        # Drain only in-flight work; never schedule the rest during server shutdown.
        with self.lock:
            for stop in self.stops.values():
                stop.set()
        self.pool.shutdown(wait=True)

    def prepared(self, rid):
        path = self.folder(rid) / "prepared.json"
        return json.loads(path.read_text()) if path.exists() else None

    def event_records(self, rid, include_unproposed=True):
        folder = self.folder(rid)
        prepared = self.prepared(rid)
        if not prepared:
            path = folder / "curated.zip"
            if not path.exists():
                return []
            with zipfile.ZipFile(path) as z:
                # Legacy runs only retain previewable conversations in the ZIP.
                ids = {
                    json.loads(line)["id"]
                    for line in z.read("dataset.jsonl").splitlines()
                }
                return [
                    r
                    for line in z.read("all-decisions.jsonl").splitlines()
                    if (r := json.loads(line))["source_conversation_id"] in ids
                ]
        results = []
        for i, detection, reverse in prepared["tasks"]:
            eid = f"{i:05d}-{int(reverse)}-{detection['turn_index']}"
            path = folder / "events" / f"{eid}.json"
            results.append(
                json.loads(path.read_text())
                if path.exists()
                else {
                    "id": eid,
                    "conversation_index": i,
                    "source_conversation_id": prepared["records"][i]["id"],
                    "target_turn_index": detection["turn_index"],
                    "roles_reversed": reverse,
                    "detection": detection,
                    "status": "pending",
                    "disposition": "needs_review",
                }
            )
        if include_unproposed:
            results.extend(prepared["unscreened"])
        return sorted(results, key=lambda e: e["id"])

    def _event(self, rid, eid):
        event = next((e for e in self.event_records(rid) if e["id"] == eid), None)
        if event is None:
            raise KeyError(eid)
        return event

    def reviews(self, rid):
        path = self.folder(rid) / "reviews.json"
        return (
            json.loads(path.read_text())
            if path.exists()
            else {"revision": 0, "items": {}, "history": []}
        )

    def review_event(self, rid, eid, decision, revision):
        if decision not in ("keep", "exclude", "unsure", "unreviewed"):
            raise ValueError("Choose Keep, Exclude or Unsure")
        with self.lock:
            self._event(rid, eid)
            value = self.reviews(rid)
            if value["revision"] != revision:
                raise Conflict(
                    "Reviews changed in another window. Refresh and try again."
                )
            previous = value["items"].get(eid, {"decision": "unreviewed"})
            value["revision"] += 1
            value["items"][eid] = {"decision": decision, "updated": now()}
            value["history"].append(
                {
                    "event_id": eid,
                    "previous": previous["decision"],
                    "decision": decision,
                    "revision": value["revision"],
                    "updated": now(),
                }
            )
            atomic_json(self.folder(rid) / "reviews.json", value)
            return value

    def resume(self, rid):
        from .runs import ACTIVE

        with self.lock:
            row = self.get(rid)
            if not self.configured:
                raise ValueError("Gemini is not configured on the server")
            if any(r["status"] in ACTIVE for r in self.list()):
                raise Conflict("A dataset is already running")
            if row["status"] not in {
                "ready",
                "paused",
                "interrupted",
                "failed",
                "completed_with_errors",
            }:
                raise Conflict("This run has no remaining work to resume")
            if (
                not self.prepared(rid)
                and not (self.folder(rid) / "upload.zip").exists()
                and not row.get("source_url")
            ):
                raise Conflict(
                    "This older run cannot resume in place. Submit its source again."
                )
            self.stops[rid] = threading.Event()
            row = self.update(
                rid,
                status="queued",
                preflight=False,
                message="Resuming remaining work",
                archived=False,
            )
            self.pool.submit(self.run, rid, row.get("source_url"))
            return row

    def pause(self, rid):
        from .runs import ACTIVE

        with self.lock:
            row = self.get(rid)
            if row["status"] not in ACTIVE or row["status"] == "packaging":
                raise Conflict("This run is not processing, or is finishing its export")
            self.stops.setdefault(rid, threading.Event()).set()
            return self.update(
                rid,
                status="pausing",
                message="Stopping new work; requests already in flight may finish",
            )

    def archive(self, rid, archived):
        from .runs import ACTIVE

        with self.lock:
            if self.get(rid)["status"] in ACTIVE:
                raise Conflict("Wait for the run to stop before archiving")
            return self.update(rid, archived=archived)

    def delete(self, rid):
        from .runs import ACTIVE

        with self.lock:
            if self.get(rid)["status"] in ACTIVE:
                raise Conflict("Wait for the run to stop before deleting")
            folder = self.folder(rid)
            shutil.rmtree(folder)
            self.stops.pop(rid, None)
        return {"deleted": True, "shared_caches_retained": True}

    def rerun(self, rid):
        from .runs import ACTIVE

        with self.lock:
            if any(r["status"] in ACTIVE for r in self.list()):
                raise Conflict("A dataset is already running")
            prepared = self.prepared(rid)
            if not prepared:
                raise Conflict("For older runs, submit the original source again")
            new_id = uuid.uuid4().hex
            dest = self.root / new_id
            dest.mkdir()
            try:
                for name in ("prepared.json", "source.json"):
                    shutil.copyfile(self.folder(rid) / name, dest / name)
                if (self.folder(rid) / "objective-plan.json").exists():
                    shutil.copyfile(
                        self.folder(rid) / "objective-plan.json",
                        dest / "objective-plan.json",
                    )
                for name in ("assets", "source-notices"):
                    shutil.copytree(self.folder(rid) / name, dest / name)
                row = {
                    **self.get(rid),
                    "id": new_id,
                    "parent_run": rid,
                    "created": now(),
                    "updated": now(),
                    "status": "ready",
                    "message": "Same validated source; ready to run. Exact model requests reuse cache.",
                    "preflight": True,
                    "processed": 0,
                    "errors": 0,
                    "selected_events": 0,
                    "selected_conversations": 0,
                    "download_ready": False,
                    "archived": False,
                }
                for key in ("zip_path", "zip_sha256", "exports"):
                    row.pop(key, None)
                atomic_json(dest / "status.json", row)
                return row
            except Exception:
                shutil.rmtree(dest)
                raise

    def _conversation(self, rid, event):
        prepared = self.prepared(rid)
        if prepared:
            return prepared["records"][event["conversation_index"]]
        with zipfile.ZipFile(self.folder(rid) / "curated.zip") as z:
            return next(
                json.loads(line)
                for line in z.read("dataset.jsonl").splitlines()
                if json.loads(line)["id"] == event["source_conversation_id"]
            )

    def event_audio(self, rid, event):
        folder = self.folder(rid)
        prepared = self.prepared(rid)
        if not prepared:
            with zipfile.ZipFile(folder / "curated.zip") as original:
                if f"clips/{event['id']}.wav" in original.namelist():
                    return self.legacy_clip(rid, event["id"]), event["evidence"]
                record = self._conversation(rid, event)
                data = original.read(record["audio"])
                if hashlib.sha256(data).hexdigest() != record["audio_sha256"]:
                    raise Conflict("Source audio integrity check failed")
                source = io.BytesIO(data)
        else:
            record = prepared["records"][event["conversation_index"]]
            source = folder / "assets" / record["file"]
            if file_hash(source) != record["audio_sha256"]:
                raise Conflict("Source audio integrity check failed")
        path = folder / "clips" / f"{event['id']}.wav"
        if path.exists() and event.get("evidence"):
            audio = path.read_bytes()
            if hashlib.sha256(audio).hexdigest() != event["evidence"]["clip_sha256"]:
                raise Conflict("Clip integrity check failed")
            return audio, event["evidence"]
        if event.get("detection", {}).get("whole_record"):
            audio = source.read_bytes()
            return audio, {
                "clip_start_s": 0,
                "clip_end_s": record["duration"],
                "target_truncated": False,
                "clip_sha256": hashlib.sha256(audio).hexdigest(),
                "source_audio_sha256": record["audio_sha256"],
            }
        # This also supports turns never proposed by the overlap detector.
        turn = record["turns"][event["target_turn_index"]]
        if turn["start_s"] is None:
            raise ValueError("This turn has no timing for audio playback")
        start = max(0, turn["start_s"] - 3)
        end = min(record["duration"], turn["end_s"] + 3, start + 30)
        with sf.SoundFile(source) as stream:
            stream.seek(int(start * stream.samplerate))
            data = stream.read(
                int(end * stream.samplerate) - int(start * stream.samplerate),
                always_2d=True,
            )
        clip = io.BytesIO()
        sf.write(clip, data, 16000, format="WAV", subtype="PCM_16")
        audio = clip.getvalue()
        return audio, {
            "clip_start_s": start,
            "clip_end_s": end,
            "target_truncated": turn["end_s"] > end,
            "clip_sha256": hashlib.sha256(audio).hexdigest(),
            "source_audio_sha256": record["audio_sha256"],
        }

    def selected_clip(self, rid, event_id):
        # The legacy route keeps its selection-only contract for existing integrations.
        if not self.get(rid)["download_ready"]:
            raise Conflict("This run has no completed selection yet")
        event = self._event(rid, event_id)
        if event.get("disposition") != "shortlist":
            raise KeyError(event_id)
        return self.event_audio(rid, event)[0]

    def detail(self, rid, offset=0, limit=10, view="selected", sample=False):
        if not 0 <= offset or not 1 <= limit <= 25:
            raise ValueError("Invalid page bounds")
        if view not in {
            "selected",
            "not_selected",
            "unresolved",
            "not_proposed",
            "all",
        }:
            raise ValueError("Unknown result filter")
        row = self.get(rid)
        reviews = self.reviews(rid)
        result = {
            "run": row,
            "feedback": self.feedback(rid),
            "reviews": reviews,
            "offset": offset,
            "limit": limit,
            "items": [],
            "total": 0,
            "coverage": row.get("coverage"),
            "legacy": not bool(self.prepared(rid)) and row.get("workflow_version") != 2,
            "reviewed_exports": self.reviewed_versions(rid),
            "sample": sample,
        }
        events = self.event_records(rid)

        def category(e):
            if e["status"] == "not_proposed":
                return "not_proposed"
            if e.get("disposition") == "shortlist":
                return "selected"
            return (
                "not_selected"
                if e.get("disposition") == "not_selected"
                else "unresolved"
            )

        result["counts"] = {
            key: sum(category(e) == key for e in events)
            for key in ("selected", "not_selected", "unresolved", "not_proposed")
        }
        # Counts remain visible at preflight, but selected previews wait for an export.
        if not row["download_ready"] and view == "selected":
            return result
        filtered = [e for e in events if view == "all" or category(e) == view]
        if sample:
            # Stable hash ordering is a reproducible pseudo-random sample, independent of verdict.
            filtered.sort(
                key=lambda e: hashlib.sha256(
                    f"review-sample-v1:{rid}:{e['id']}".encode()
                ).digest()
            )
        result["total"] = len(filtered)
        for event in filtered[offset : offset + limit]:
            record = self._conversation(rid, event)
            target = event["target_turn_index"]
            start = max(0, target - 10)
            turns = record["turns"][start : target + 11]
            metadata = event.get("evidence")
            if not metadata and event.get("detection", {}).get("whole_record"):
                metadata = {"clip_start_s": 0, "clip_end_s": record["duration"]}
            if not metadata:
                turn = record["turns"][target]
                onset = max(0, (turn["start_s"] or 0) - 3)
                metadata = {
                    "clip_start_s": onset,
                    "clip_end_s": min(
                        record.get("duration", turn["end_s"] or 0),
                        (turn["end_s"] or 0) + 3,
                        onset + 30,
                    ),
                }
            result["items"].append(
                {
                    **event,
                    "conversation_id": record["id"],
                    "evidence": metadata,
                    "category": category(event),
                    "review": reviews["items"].get(
                        event["id"], {"decision": "unreviewed"}
                    ),
                    "answer": event.get(
                        "answer",
                        {
                            "evidence_note": "Not proposed by the overlap detector"
                            if category(event) == "not_proposed"
                            else "Analysis incomplete; no negative label assigned"
                        },
                    ),
                    "turns": [
                        {**t, "index": start + i, "text": t["text"][:2000]}
                        for i, t in enumerate(turns)
                    ],
                    "context_truncated": start > 0
                    or target + 11 < len(record["turns"])
                    or any(len(t["text"]) > 2000 for t in turns),
                }
            )
        return result

    def reviewed_versions(self, rid):
        path = self.folder(rid) / "reviewed-exports.json"
        return json.loads(path.read_text()) if path.exists() else []

    def reviewed_export(self, rid, revision):
        from .runs import ACTIVE

        with self.lock:
            if self.get(rid)["status"] in ACTIVE:
                raise Conflict("Wait for the run to stop before exporting reviews")
            reviews = self.reviews(rid)
            if reviews["revision"] != revision:
                raise Conflict("Reviews changed. Refresh before exporting")
            if not revision:
                raise Conflict("Review at least one example first")
            versions = self.reviewed_versions(rid)
            existing = next(
                (v for v in versions if v["review_revision"] == revision), None
            )
            if existing:
                return existing
            selected = [
                e
                for e in self.event_records(rid)
                if reviews["items"].get(e["id"], {}).get("decision") == "keep"
            ]
            eid = uuid.uuid4().hex
            target = self.folder(rid) / f"reviewed-{eid}.zip"
            prepared = self.prepared(rid)
            manifest = {
                "run_id": rid,
                "requirement": self.get(rid)["requirement"],
                "objective_plan": self.get(rid).get("objective_plan"),
                "policy_version": self.get(rid).get("policy_version"),
                "export_id": eid,
                "review_revision": revision,
                "created": now(),
                "label_status": "human_reviewed_keep",
                "selected_events": len(selected),
                "selected_conversations": len(
                    {e["source_conversation_id"] for e in selected}
                ),
                "source_provenance": prepared["provenance"]
                if prepared
                else {"legacy_run": rid},
                "coverage": self.get(rid).get("coverage"),
                "run_status": self.get(rid)["status"],
                "selection": "Only explicitly kept events. Unreviewed, unsure and excluded events are not positive labels. Full conversations may contain other, unlabeled speech.",
            }
            self.write_bundle(rid, target, selected, manifest, reviewed=reviews)
            version = {
                "id": eid,
                "review_revision": revision,
                "created": manifest["created"],
                "selected_events": len(selected),
                "sha256": file_hash(target),
            }
            versions.append(version)
            atomic_json(self.folder(rid) / "reviewed-exports.json", versions)
            return version

    def reviewed_download(self, rid, eid):
        version = next((v for v in self.reviewed_versions(rid) if v["id"] == eid), None)
        if not version:
            raise KeyError(eid)
        path = self.folder(rid) / f"reviewed-{eid}.zip"
        if file_hash(path) != version["sha256"]:
            raise Conflict("Reviewed export integrity check failed")
        return path

    def package(self, rid, prepared, results, status):
        selected = [e for e in results if e.get("disposition") == "shortlist"]
        errors = sum(e["status"] == "error" for e in results)
        unresolved = sum(e.get("disposition") == "needs_review" for e in results)
        fields = {
            "status": status,
            "errors": errors,
            "unresolved": unresolved,
            "selected_events": len(selected),
            "selected_conversations": len(
                {e["source_conversation_id"] for e in selected}
            ),
            "message": "Paused; resume remaining work when ready"
            if status == "paused"
            else ("Finished with unresolved analysis errors" if errors else "Finished"),
        }
        if status == "paused":
            if any(e.get("error") == "RateLimit" for e in results):
                fields["message"] = (
                    "Gemini rate limit reached. Completed results are saved; wait before resuming."
                )
            elif any(e.get("error") == "ProviderAccess" for e in results):
                fields["message"] = (
                    "Gemini access failed. Completed results are saved; check model access before resuming."
                )
        manifest = {
            **self.get(rid),
            **fields,
            "source_provenance": prepared["provenance"],
            "label_status": "model_selected_not_human_verified",
            "download_ready": True,
            "coverage": prepared["coverage"],
            "untimed_turns": prepared["coverage"]["untimed_turns"],
            "event_errors": [r["id"] for r in results if r["status"] == "error"],
            "pending_events": [r["id"] for r in results if r["status"] == "pending"],
            "selection": (
                "Complete recordings clearly matching the saved objective and rubric. Unclear judgments and errors are not negatives. No accuracy guarantee."
                if prepared.get("mode") == "audio"
                else "At least one successful-interruption-v1 shortlist event. Overlap proposes candidates; no recall guarantee."
            ),
        }
        name = f"model-{uuid.uuid4().hex}.zip"
        target = self.folder(rid) / name
        self.write_bundle(rid, target, selected, manifest, results=results)
        version = {
            "path": name,
            "sha256": file_hash(target),
            "created": now(),
            "status": status,
        }
        self.update(
            rid,
            **fields,
            download_ready=True,
            zip_path=name,
            zip_sha256=version["sha256"],
            exports=self.get(rid).get("exports", []) + [version],
        )

    def write_bundle(
        self, rid, target, selected, manifest, results=None, reviewed=None
    ):
        folder = self.folder(rid)
        prepared = self.prepared(rid)
        rows = []
        with zipfile.ZipFile(
            target.with_suffix(".tmp"), "w", compression=zipfile.ZIP_DEFLATED
        ) as z:
            for cid in sorted({e["source_conversation_id"] for e in selected}):
                record = self._conversation(
                    rid, next(e for e in selected if e["source_conversation_id"] == cid)
                )
                row = {
                    k: v
                    for k, v in record.items()
                    if k not in ("file", "audio", "both_directions", "curation")
                }
                audio_name = f"audio/{record['file']}" if prepared else record["audio"]
                row.update(
                    audio=audio_name,
                    curation={
                        "label_status": manifest["label_status"],
                        "events": [
                            e["id"]
                            for e in selected
                            if e["source_conversation_id"] == cid
                        ],
                    },
                )
                rows.append(row)
                if prepared:
                    if (
                        file_hash(folder / "assets" / record["file"])
                        != record["audio_sha256"]
                    ):
                        raise Conflict("Source audio integrity check failed")
                    z.write(folder / "assets" / record["file"], audio_name)
                else:
                    with zipfile.ZipFile(folder / "curated.zip") as original:
                        z.writestr(audio_name, original.read(audio_name))
            annotations = []
            for event in selected:
                audio, metadata = self.event_audio(rid, event)
                z.writestr(f"clips/{event['id']}.wav", audio)
                if reviewed:
                    # Model verdicts are historical evidence, not reviewed positive annotations.
                    annotations.append(
                        {
                            "id": event["id"],
                            "source_conversation_id": event["source_conversation_id"],
                            "target_turn_index": event["target_turn_index"],
                            "roles_reversed": event["roles_reversed"],
                            "evidence": metadata,
                            "review": reviewed["items"][event["id"]],
                            "label_status": "human_reviewed_keep",
                        }
                    )
                else:
                    annotations.append(event)
            z.writestr("dataset.jsonl", "".join(canonical(r) + "\n" for r in rows))
            z.writestr(
                "selected-events.jsonl",
                "".join(canonical(r) + "\n" for r in annotations),
            )
            if reviewed:
                z.writestr("review-audit.json", canonical(reviewed))
            else:
                z.writestr(
                    "all-decisions.jsonl",
                    "".join(canonical(r) + "\n" for r in results or []),
                )
            z.writestr("manifest.json", json.dumps(manifest, indent=2))
            z.writestr(
                "README.txt",
                manifest["selection"]
                + "\nLabel status: "
                + manifest["label_status"]
                + "\nSource licenses apply. Full conversations are retained; only listed events have positive annotations.\nSee manifest.json for coverage and completion.\n",
            )
            if prepared:
                for p in sorted((folder / "source-notices").iterdir()):
                    z.write(p, f"source-notices/{p.name}")
            else:
                with zipfile.ZipFile(folder / "curated.zip") as original:
                    for name in original.namelist():
                        if name.startswith("source-notices/"):
                            z.writestr(name, original.read(name))
        target.with_suffix(".tmp").replace(target)
