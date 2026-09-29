from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import soxr

RATE = 24000


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class EventLog:
    """Append-only events on one virtual clock, linked by content hashes."""

    def __init__(self):
        self.events: list[dict] = []

    def emit(self, sample: int, kind: str, payload: dict, parents=()) -> dict:
        if type(sample) is not int or sample < 0:
            raise ValueError("sample must be a nonnegative integer")
        if self.events and sample < self.events[-1]["sample"]:
            raise ValueError("clock reversal")
        if any(type(p) is not int or p < 0 or p >= len(self.events) for p in parents):
            raise ValueError("unknown causal parent")
        event = {
            "schema_version": 1,
            "seq": len(self.events),
            "clock_id": "virtual_media",
            "sample": sample,
            "sample_rate": RATE,
            "kind": kind,
            "parents": list(parents),
            "payload": json.loads(canonical(payload)),
            "previous_hash": self.events[-1]["hash"] if self.events else None,
        }
        event["hash"] = hashlib.sha256(canonical(event).encode()).hexdigest()
        self.events.append(event)
        return event

    def save(self, path: Path):
        with path.open("x") as f:
            for event in self.events:
                f.write(canonical(event) + "\n")

    @classmethod
    def read(cls, path: Path):
        log = cls()
        for line in path.read_text().splitlines():
            e = json.loads(line)
            actual = log.emit(e["sample"], e["kind"], e["payload"], e["parents"])
            if actual != e:
                raise ValueError("corrupt event chain or schema")
        return log


class Scheduler:
    """Stable ordering for tied deadlines; scheduling into the past is forbidden."""

    def __init__(self):
        self.now = 0
        self._queue = []
        self._sequence = 0

    def at(self, sample: int, action: Callable):
        if type(sample) is not int or sample < self.now:
            raise ValueError("invalid deadline")
        heapq.heappush(self._queue, (sample, self._sequence, action))
        self._sequence += 1

    def run(self):
        while self._queue:
            self.now, _, action = heapq.heappop(self._queue)
            action()


@dataclass
class Segment:
    id: str
    epoch: int
    pcm: np.ndarray
    consumed: int = 0


class Playback:
    """Simulated renderer: generated, enqueued, and consumed audio remain distinct."""

    def __init__(self, log: EventLog, capacity_samples=RATE * 2):
        self.log = log
        self.capacity = capacity_samples
        self.epoch = 0
        self.queue: deque[Segment] = deque()
        self.ids = set()
        self.rendered: list[dict] = []

    @property
    def queued(self):
        return sum(len(s.pcm) - s.consumed for s in self.queue)

    def enqueue(self, sample: int, id: str, epoch: int, pcm) -> bool:
        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1 or len(pcm) == 0 or not np.isfinite(pcm).all():
            raise ValueError("invalid mono PCM")
        if id in self.ids:
            raise ValueError("duplicate segment")
        self.ids.add(id)
        if epoch != self.epoch:
            self.log.emit(
                sample,
                "audio.discarded",
                {
                    "id": id,
                    "epoch": epoch,
                    "reason": "stale_epoch",
                    "samples": len(pcm),
                },
            )
            return False
        if self.queued + len(pcm) > self.capacity:
            self.log.emit(
                sample, "audio.rejected", {"id": id, "reason": "backpressure"}
            )
            return False
        self.queue.append(Segment(id, epoch, pcm.copy()))
        self.log.emit(
            sample, "audio.enqueued", {"id": id, "epoch": epoch, "samples": len(pcm)}
        )
        return True

    def cancel(self, sample: int):
        discarded = [
            {"id": s.id, "remaining": len(s.pcm) - s.consumed} for s in self.queue
        ]
        self.queue.clear()
        self.epoch += 1
        self.log.emit(
            sample,
            "playback.cancelled",
            {"new_epoch": self.epoch, "discarded": discarded},
        )

    def render(self, sample: int, count: int):
        if type(count) is not int or count <= 0:
            raise ValueError("invalid render size")
        out = np.zeros(count, dtype=np.float32)
        cursor = 0
        while cursor < count and self.queue:
            s = self.queue[0]
            n = min(count - cursor, len(s.pcm) - s.consumed)
            out[cursor : cursor + n] = s.pcm[s.consumed : s.consumed + n]
            item = {
                "id": s.id,
                "epoch": s.epoch,
                "output_start": sample + cursor,
                "source_start": s.consumed,
                "samples": n,
            }
            self.rendered.append(item)
            self.log.emit(sample, "audio.rendered", item)
            cursor += n
            s.consumed += n
            if s.consumed == len(s.pcm):
                self.queue.popleft()
        if cursor < count:
            self.log.emit(
                sample,
                "playback.silence",
                {"start": sample + cursor, "samples": count - cursor},
            )
        return out


class TranscriptHistory:
    """Revision history preserves availability, not just retrospective word time."""

    def __init__(self):
        self.items = []

    def add(
        self, utterance: str, revision: int, text: str, audio_end: int, available: int
    ):
        if any(type(x) is not int or x < 0 for x in (revision, audio_end, available)):
            raise ValueError("revision and times must be nonnegative integers")
        if available < audio_end or min(audio_end, revision) < 0:
            raise ValueError("future audio or negative revision")
        prior = [x for x in self.items if x["utterance"] == utterance]
        if prior and (
            revision <= prior[-1]["revision"] or available < prior[-1]["available"]
        ):
            raise ValueError("revision/availability reversal")
        self.items.append(
            dict(
                utterance=utterance,
                revision=revision,
                text=text,
                audio_end=audio_end,
                available=available,
            )
        )

    def visible(self, now: int):
        latest = {}
        for item in self.items:
            if item["available"] <= now:
                latest[item["utterance"]] = dict(item)
        return latest


class State:
    def __init__(self, values: dict):
        self.values = dict(values)
        self.version = 0
        self.epoch = 0

    def cancel(self):
        self.epoch += 1

    def patch(self, changes: dict, version: int, epoch: int):
        if version != self.version or epoch != self.epoch:
            raise ValueError("stale proposal")
        if not changes.keys() <= self.values.keys():
            raise ValueError("unknown fields")
        self.values.update(changes)
        self.version += 1


def public_scenario(scenario: dict):
    # Allowlist rather than removing a few known answer keys.
    keys = ("scenario_id", "initial_utterance", "intervention", "schedule")
    return json.loads(canonical({k: scenario[k] for k in keys if k in scenario}))


def streaming_resample(chunks, input_rate: int, output_rate: int):
    stream = soxr.ResampleStream(
        input_rate, output_rate, 1, dtype="float32", quality="HQ"
    )
    out = []
    for chunk in chunks:
        a = np.asarray(chunk, dtype=np.float32)
        if a.ndim != 1 or not np.isfinite(a).all():
            raise ValueError("invalid audio")
        if len(a):
            out.append(stream.resample_chunk(a))
    out.append(stream.resample_chunk(np.empty(0, dtype=np.float32), last=True))
    return np.concatenate(out)


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if not (math.isfinite(a) and math.isfinite(b)) or a < 0 or b < a:
            raise ValueError("invalid interval")
        if a == b:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def overlap(a, b):
    return sum(max(0, min(y, v) - max(x, u)) for x, y in union(a) for u, v in union(b))


def yield_delay(agent, onset: int, horizon: int, silence: int):
    if horizon <= onset or silence <= 0:
        raise ValueError("invalid observation window")
    intervals = union(agent)
    if not any(a <= onset < b for a, b in intervals):
        return {"status": "not_applicable", "samples": None}
    cursor = onset
    for a, b in intervals:
        if b <= onset:
            continue
        if a - cursor >= silence and cursor + silence <= horizon:
            return {"status": "observed", "samples": cursor - onset}
        cursor = max(cursor, b)
    if cursor + silence <= horizon:
        return {"status": "observed", "samples": cursor - onset}
    return {"status": "right_censored", "samples": horizon - onset}
