"""SQLite run index plus durable, hash-linked host-clock evidence."""

import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from .core import canonical

TERMINAL = {"complete", "failed", "aborted", "interrupted"}


def atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Journal:
    def __init__(self, path, run_id):
        self.path = path
        self.run_id = run_id
        self.seq = 0
        self.previous = None
        self.start = time.monotonic_ns()
        self.file = path.open("x")

    def emit(self, kind, payload, parents=()):
        if any(type(p) is not int or p < 0 or p >= self.seq for p in parents):
            raise ValueError("unknown parent")
        event = {
            "schema_version": 1,
            "run_id": self.run_id,
            "seq": self.seq,
            "kind": kind,
            "clock_id": "host_monotonic",
            "observed_at_ns": time.monotonic_ns(),
            "relative_ns": time.monotonic_ns() - self.start,
            "parents": list(parents),
            "payload": payload,
            "previous_hash": self.previous,
        }
        event["hash"] = hashlib.sha256(canonical(event).encode()).hexdigest()
        self.file.write(canonical(event) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.seq += 1
        self.previous = event["hash"]
        return event["seq"]

    def close(self):
        self.file.close()


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created TEXT, status TEXT, scenario TEXT, error TEXT)"
            )
            # Never pretend a process restart preserved a running model conversation.
            db.execute(
                "UPDATE runs SET status='interrupted', error='Server stopped before finalization; start a new attempt.' WHERE status NOT IN ('complete','failed','aborted','interrupted')"
            )

    def connect(self):
        db = sqlite3.connect(self.root / "index.sqlite")
        db.row_factory = sqlite3.Row
        return db

    def create(self, scenario):
        run = uuid.uuid4().hex
        (self.root / run).mkdir()
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?)",
                (run, datetime.now(timezone.utc).isoformat(), "queued", scenario, None),
            )
        return run

    def update(self, run, status, error=None):
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET status=?, error=? WHERE id=?", (status, error, run)
            )

    def get(self, run):
        with self.connect() as db:
            r = db.execute("SELECT * FROM runs WHERE id=?", (run,)).fetchone()
        if r is None:
            raise KeyError(run)
        return dict(r)

    def list(self):
        with self.connect() as db:
            return [
                dict(r) for r in db.execute("SELECT * FROM runs ORDER BY created DESC")
            ]

    def folder(self, run):
        self.get(run)
        return self.root / run

    def detail(self, run):
        value = self.get(run)
        path = self.folder(run) / "manifest.json"
        value["manifest"] = json.loads(path.read_text()) if path.exists() else None
        return value

    def events(self, run):
        path = self.folder(run) / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines(keepends=True)
        return [json.loads(line) for line in lines if line.endswith("\n")]

    def verify(self, run):
        data = self.detail(run)
        if data["status"] != "complete" or data["manifest"] is None:
            raise ValueError("run is not complete")
        m = data["manifest"]
        folder = self.folder(run)
        for name, digest in m["artifact_hashes"].items():
            if Path(name).name != name or sha(folder / name) != digest:
                raise ValueError("artifact mismatch")
        previous = None
        for seq, e in enumerate(self.events(run)):
            given = e.pop("hash")
            if (
                e["seq"] != seq
                or e["previous_hash"] != previous
                or hashlib.sha256(canonical(e).encode()).hexdigest() != given
            ):
                raise ValueError("event chain mismatch")
            previous = given
        if previous != m["last_event_hash"]:
            raise ValueError("final event mismatch")
        return {"verified": True, "artifacts": len(m["artifact_hashes"])}
