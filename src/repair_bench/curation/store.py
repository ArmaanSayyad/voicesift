import hashlib, json, sqlite3, uuid
from pathlib import Path
from datetime import datetime, timezone
from ..core import canonical
from ..store import atomic_json
from .models import Conversation, Review
from .detect import detect, VERSION


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class Conflict(ValueError):
    pass


class Corpus:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "assets").mkdir(exist_ok=True)
        (self.root / "exports").mkdir(exist_ok=True)
        with self.db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY, external_id TEXT, group_id TEXT, split TEXT, hash TEXT UNIQUE, body TEXT, created TEXT);
            CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, conversation_id TEXT, turn_index INTEGER, UNIQUE(conversation_id,turn_index));
            CREATE TABLE IF NOT EXISTS reviews(seq INTEGER PRIMARY KEY AUTOINCREMENT,candidate_id TEXT,version INTEGER,body TEXT,created TEXT,UNIQUE(candidate_id,version));
            CREATE TABLE IF NOT EXISTS assets(conversation_id TEXT PRIMARY KEY,name TEXT,hash TEXT,duration REAL);
            CREATE TABLE IF NOT EXISTS exports(id TEXT PRIMARY KEY,manifest TEXT,created TEXT);
            """)

    def db(self):
        db = sqlite3.connect(self.root / "corpus.sqlite", timeout=20)
        db.row_factory = sqlite3.Row
        return db

    def import_jsonl(self, text):
        lines = [x for x in text.splitlines() if x.strip()]
        if not 1 <= len(lines) <= 100:
            raise ValueError("Import 1–100 JSONL conversations per batch")
        parsed = [Conversation.model_validate_json(x).model_dump() for x in lines]
        if sum(len(x["turns"]) for x in parsed) > 2000:
            raise ValueError("At most 2000 turns per batch")
        added = duplicates = candidates = 0
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            for body in parsed:
                group = body["source_group"]
                split = body["split"]
                existing = db.execute(
                    "SELECT split FROM conversations WHERE group_id=? LIMIT 1", (group,)
                ).fetchone()
                if existing and existing["split"] != split:
                    raise Conflict("A source group cannot cross train/dev/test splits")
                h = digest(
                    [
                        {
                            "role": x["role"],
                            "text": " ".join(x["text"].lower().split()),
                            "start_s": x["start_s"],
                            "end_s": x["end_s"],
                        }
                        for x in body["turns"]
                    ]
                )
                match = db.execute(
                    "SELECT split FROM conversations WHERE hash=?", (h,)
                ).fetchone()
                if match:
                    if match["split"] != split:
                        raise Conflict("Duplicate content cannot cross splits")
                    duplicates += 1
                    continue
                cid = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO conversations VALUES (?,?,?,?,?,?,?)",
                    (cid, body["id"], group, split, h, canonical(body), now()),
                )
                overlaps, _ = detect(body)
                for overlap in overlaps:
                    db.execute(
                        "INSERT INTO candidates VALUES (?,?,?)",
                        (uuid.uuid4().hex, cid, overlap["turn_index"]),
                    )
                    candidates += 1
                added += 1
            if db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] > 1000:
                raise ValueError(
                    "This pilot supports at most 1000 overlap candidates; import a smaller source subset"
                )
        return {"imported": added, "duplicates": duplicates, "candidates": candidates}

    def ids(self):
        with self.db() as db:
            return [
                x[0] for x in db.execute("SELECT id FROM candidates ORDER BY rowid")
            ]

    def get(self, candidate):
        with self.db() as db:
            row = db.execute(
                "SELECT c.*,s.body,s.hash FROM candidates c JOIN conversations s ON s.id=c.conversation_id WHERE c.id=?",
                (candidate,),
            ).fetchone()
            if not row:
                raise KeyError(candidate)
            result = dict(row)
            result["conversation"] = json.loads(result.pop("body"))
            result["detection"] = next(
                x
                for x in detect(result["conversation"])[0]
                if x["turn_index"] == result["turn_index"]
            )
            r = db.execute(
                "SELECT * FROM reviews WHERE candidate_id=? ORDER BY version DESC LIMIT 1",
                (candidate,),
            ).fetchone()
            result["review"] = (
                None
                if not r
                else {
                    **json.loads(r["body"]),
                    "version": r["version"],
                    "created": r["created"],
                }
            )
            asset = db.execute(
                "SELECT * FROM assets WHERE conversation_id=?",
                (row["conversation_id"],),
            ).fetchone()
            result["audio"] = dict(asset) if asset else None
            return result

    def list(self):
        # Bounded work queue; full corpus is preserved in SQLite.
        ids = self.ids()
        return {
            "total": len(ids),
            "items": [self.get(i) for i in ids[:1000]],
            "limit": 1000,
        }

    def review(self, candidate, review: Review):
        self.get(candidate)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            r = (
                db.execute(
                    "SELECT MAX(version) FROM reviews WHERE candidate_id=?",
                    (candidate,),
                ).fetchone()[0]
                or 0
            )
            if r != review.expected_version:
                raise Conflict("Review changed in another tab; reload before saving")
            body = review.model_dump()
            body.pop("expected_version")
            db.execute(
                "INSERT INTO reviews(candidate_id,version,body,created) VALUES (?,?,?,?)",
                (candidate, r + 1, canonical(body), now()),
            )
        return self.get(candidate)

    def metrics(self):
        labels = {}
        included = 0
        timed = untimed = conversations = 0
        with self.db() as db:
            for row in db.execute("SELECT body FROM conversations"):
                conversations += 1
                _, coverage = detect(json.loads(row["body"]))
                timed += coverage["timed_user_turns"]
                untimed += coverage["untimed_user_turns"]
        for cid in self.ids():
            r = self.get(cid)["review"]
            if r:
                labels[r["label"]] = labels.get(r["label"], 0) + 1
                included += int(r["include"])
        return {
            "conversations": conversations,
            "candidates": len(self.ids()),
            "reviewed": sum(labels.values()),
            "included": included,
            "labels": labels,
            "timed_user_turns": timed,
            "untimed_user_turns": untimed,
            "detector": VERSION,
            "scope": "Timestamp overlap candidates; no accuracy or recall claim. Untimed turns are not searched.",
        }

    def export(self):
        rows = []
        for cid in self.ids():
            c = self.get(cid)
            r = c["review"]
            if r and r["include"] and r["label"] == "interruption":
                if c["audio"]:
                    asset = self.root / "assets" / c["audio"]["name"]
                    if (
                        not asset.exists()
                        or hashlib.sha256(asset.read_bytes()).hexdigest()
                        != c["audio"]["hash"]
                    ):
                        raise ValueError("Source audio hash mismatch; export refused")
                rows.append(
                    {
                        "schema_version": 1,
                        "candidate_id": cid,
                        "conversation": c["conversation"],
                        "source_content_hash": c["hash"],
                        "target_turn_index": c["turn_index"],
                        "annotation": r,
                        "detection": c["detection"],
                        "audio": c["audio"],
                        "scope": "Human-annotated interruption candidate; no correction/cancellation or acoustic-quality label implied.",
                    }
                )
        if not rows:
            raise ValueError(
                "No explicitly included, reviewer-confirmed interruptions to export"
            )
        eid = uuid.uuid4().hex
        path = self.root / "exports" / f"{eid}.jsonl"
        with path.open("x") as f:
            for row in rows:
                f.write(canonical(row) + "\n")
        manifest = {
            "id": eid,
            "records": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "created": now(),
            "audio_policy": "References and hashes only; original WAVs remain in the local asset store.",
        }
        atomic_json(path.with_suffix(".manifest.json"), manifest)
        with self.db() as db:
            db.execute(
                "INSERT INTO exports VALUES (?,?,?)",
                (eid, canonical(manifest), manifest["created"]),
            )
        return manifest
