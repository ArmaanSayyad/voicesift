import asyncio
import hashlib
import io
import json
import secrets
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .analysis import Analysis, evidence
from .models import Analyze, Import, Review
from .precision import VERSION as POLICY_VERSION
from .runs import Runs
from .sources import MAX_UPLOAD, REQUIREMENT
from .store import Conflict, Corpus


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=10000, strict=True)


class DatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str
    requirement: str = REQUIREMENT
    preflight: bool = False


class EventReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["keep", "exclude", "unsure", "unreviewed"]
    revision: int = Field(ge=0)


class ExportReviewRequest(BaseModel):
    revision: int = Field(ge=0)


class ArchiveRequest(BaseModel):
    archived: bool


REPO = Path(__file__).resolve().parents[3]
MAX_AUDIO = 50_000_000


def create_app(root=None, backend=None):
    corpus = Corpus(root or REPO / "artifacts/interruption-curation")
    analysis = Analysis(corpus, backend)
    runs = Runs(
        corpus.root / "dataset-runs", backend=backend, configured=analysis.configured
    )
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        yield
        await asyncio.to_thread(runs.shutdown)

    app = FastAPI(title="Interruption Curation", lifespan=lifespan)
    app.state.corpus = corpus
    app.state.analysis = analysis
    app.state.runs = runs
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if request.headers.get("origin") not in (
                None,
                str(request.base_url).rstrip("/"),
            ) or not secrets.compare_digest(
                request.headers.get("x-repair-token", ""), token
            ):
                return JSONResponse(
                    {"detail": "Local session token and matching origin required"},
                    status_code=403,
                )
            if request.url.path != "/api/curation/runs/upload":
                limit = MAX_AUDIO if request.url.path.endswith("/audio") else 2_100_000
                total = 0
                chunks = []
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > limit:
                        return JSONResponse(
                            {"detail": "Upload too large"}, status_code=413
                        )
                    chunks.append(chunk)
                request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; connect-src 'self'; frame-ancestors 'none'",
            }
        )
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Record not found"}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse(
            {"detail": str(exc)}, status_code=409 if isinstance(exc, Conflict) else 422
        )

    @app.exception_handler(ValidationError)
    async def validation(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/api/curation/bootstrap")
    def bootstrap():
        return {
            "token": token,
            "goal": "user onset during agent speech",
            "laya_enabled": False,
            "gemini_configured": analysis.configured,
            "policy_version": POLICY_VERSION,
            "requirement": REQUIREMENT,
        }

    @app.get("/api/curation/format", response_class=PlainTextResponse)
    def dataset_format():
        return """Upload format

A ZIP containing dataset.jsonl at its root and the referenced audio files.
One JSON object per line, one conversation per object. Example:

{"id":"call-001","audio":"audio/call-001.wav","turns":[{"role":"assistant","text":"The first option is...","start_s":0.0,"end_s":3.0},{"role":"user","text":"Wait, let me clarify.","start_s":1.5,"end_s":4.0}]}

This example shows the structure, not a validated interruption.
Use actual mono/stereo audio (up to one hour), speaker labels, transcripts and timestamps in seconds.
User-on-assistant interruptions are searched. Raw audio without timed transcripts is not supported yet.
Optional: source_group, split, provenance, timing_source. Include source LICENSE/README.md notices.
ZIP: up to 512 MB compressed / 2 GB expanded. 1–500 conversations, up to 10,000 candidate events per run.

Hugging Face
Use https://huggingface.co/datasets/owner/name. The repo must contain dataset.jsonl plus its audio files,
or use mundo-ai/turn-benchmark-dev (adapter included, both speaker directions searched).
Up to 6 GB of selected source files. Gated sources require accepted terms and local Hugging Face CLI login.
Unsupported schemas are rejected; dataset code is never executed.

Results
The ZIP contains selected full conversations and 16 kHz WAV audio, event clips, model decisions,
source notices and a manifest. These are model-selected candidates, not human-confirmed labels.
"""

    @app.get("/api/curation/runs")
    def dataset_runs():
        return {"items": runs.list()}

    @app.post("/api/curation/runs/huggingface")
    def dataset_link(body: DatasetRequest):
        return runs.submit(body.requirement, url=body.url, preflight=body.preflight)

    @app.post("/api/curation/runs/upload")
    async def dataset_upload(
        request: Request,
        filename: str = "dataset.zip",
        requirement: str = REQUIREMENT,
        preflight: bool = False,
    ):
        if requirement != REQUIREMENT:
            raise ValueError("Only the fixed interruption requirement is supported")
        if not runs.configured and not preflight:
            raise ValueError("Gemini is not configured on the server")
        if not filename.lower().endswith(".zip"):
            raise ValueError(
                "Upload a ZIP containing dataset.jsonl and its referenced audio"
            )
        path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=runs.root, suffix=".zip", delete=False
            ) as f:
                path = Path(f.name)
                size = 0
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "Upload exceeds 512 MB")
                    f.write(chunk)
            return runs.submit(
                requirement, upload=path, filename=filename, preflight=preflight
            )
        finally:
            if path:
                path.unlink(missing_ok=True)

    @app.get("/api/curation/runs/{rid}")
    def dataset_detail(
        rid: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(10, ge=1, le=25),
        view: str = "selected",
        sample: bool = False,
    ):
        return runs.detail(rid, offset, limit, view, sample)

    @app.put("/api/curation/runs/{rid}/feedback")
    def dataset_feedback(rid: str, body: FeedbackRequest):
        return runs.save_feedback(rid, body.text)

    @app.get("/api/curation/runs/{rid}/clips/{event_id}")
    def dataset_clip(rid: str, event_id: str):
        return Response(runs.selected_clip(rid, event_id), media_type="audio/wav")

    @app.get("/api/curation/runs/{rid}/download")
    def dataset_download(rid: str):
        return FileResponse(
            runs.download(rid),
            media_type="application/zip",
            filename=f"curated-{rid[:8]}.zip",
        )

    @app.get("/api/curation/runs/{rid}/model-exports/{name}")
    def model_download(rid: str, name: str):
        row = runs.get(rid)
        version = next((v for v in row.get("exports", []) if v["path"] == name), None)
        if not version:
            raise KeyError(name)
        path = runs.folder(rid) / name
        from .sources import file_hash

        if file_hash(path) != version["sha256"]:
            raise Conflict("Export integrity check failed")
        return FileResponse(path, media_type="application/zip", filename=name)

    @app.post("/api/curation/runs/{rid}/resume")
    def resume_run(rid: str):
        return runs.resume(rid)

    @app.post("/api/curation/runs/{rid}/pause")
    def pause_run(rid: str):
        return runs.pause(rid)

    @app.post("/api/curation/runs/{rid}/rerun")
    def rerun(rid: str):
        return runs.rerun(rid)

    @app.put("/api/curation/runs/{rid}/archive")
    def archive_run(rid: str, body: ArchiveRequest):
        return runs.archive(rid, body.archived)

    @app.delete("/api/curation/runs/{rid}")
    def delete_run(rid: str):
        return runs.delete(rid)

    @app.put("/api/curation/runs/{rid}/reviews/{eid}")
    def review_event(rid: str, eid: str, body: EventReviewRequest):
        return runs.review_event(rid, eid, body.decision, body.revision)

    @app.get("/api/curation/runs/{rid}/events/{eid}/audio")
    def event_audio(rid: str, eid: str):
        return Response(
            runs.event_audio(rid, runs._event(rid, eid))[0], media_type="audio/wav"
        )

    @app.post("/api/curation/runs/{rid}/reviewed-exports")
    def reviewed_export(rid: str, body: ExportReviewRequest):
        return runs.reviewed_export(rid, body.revision)

    @app.get("/api/curation/runs/{rid}/reviewed-exports/{eid}")
    def reviewed_download(rid: str, eid: str):
        return FileResponse(
            runs.reviewed_download(rid, eid),
            media_type="application/zip",
            filename=f"reviewed-{rid[:8]}-{eid[:8]}.zip",
        )

    @app.get("/api/curation/example.zip")
    def example_zip():
        audio = io.BytesIO()
        sf.write(audio, np.zeros(80000), 16000, format="WAV", subtype="PCM_16")
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr(
                "dataset.jsonl",
                json.dumps(
                    {
                        "id": "format-example",
                        "audio": "audio/example.wav",
                        "provenance": "Silent structural fixture, not speech or a validated interruption",
                        "turns": [
                            {
                                "role": "assistant",
                                "text": "Replace with actual transcript",
                                "start_s": 0,
                                "end_s": 3,
                            },
                            {
                                "role": "user",
                                "text": "Replace with actual transcript",
                                "start_s": 1,
                                "end_s": 4,
                            },
                        ],
                    }
                )
                + "\n",
            )
            z.writestr("audio/example.wav", audio.getvalue())
            z.writestr(
                "README.md",
                "Format template only. The audio is SILENCE and the timestamps are invented. Replace audio and transcripts with your real data before curation.\n",
            )
        return Response(
            output.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="dataset-format-example.zip"'
            },
        )

    @app.get("/api/curation/jobs")
    def jobs():
        return {"items": analysis.jobs()}

    @app.post("/api/curation/analyze")
    def analyze(body: Analyze):
        return analysis.submit(body.candidate_ids)

    @app.get("/api/curation/candidates/{candidate}/clip")
    def clip(candidate: str):
        _, audio, _ = evidence(corpus, candidate)
        return Response(audio, media_type="audio/wav")

    @app.get("/api/curation/candidates")
    def candidates():
        return corpus.list()

    @app.get("/api/curation/metrics")
    def metrics():
        return corpus.metrics()

    @app.get("/api/curation/example")
    def example():
        return FileResponse(
            REPO / "fixtures/interruption-demo.jsonl",
            media_type="application/x-ndjson",
            filename="interruption-demo.jsonl",
        )

    @app.post("/api/curation/import")
    def ingest(body: Import):
        return corpus.import_jsonl(body.jsonl)

    @app.post("/api/curation/reviews/{candidate}")
    def review(candidate: str, body: Review):
        return corpus.review(candidate, body)

    @app.post("/api/curation/exports")
    def export():
        return corpus.export()

    @app.get("/api/curation/exports/{eid}")
    def download(eid: str):
        with corpus.db() as db:
            r = db.execute("SELECT manifest FROM exports WHERE id=?", (eid,)).fetchone()
        if not r:
            raise KeyError(eid)
        manifest = json.loads(r["manifest"])
        path = corpus.root / "exports" / f"{eid}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
            raise HTTPException(409, "Export hash mismatch")
        return FileResponse(
            path,
            media_type="application/x-ndjson",
            filename=f"interruptions-{eid}.jsonl",
        )

    @app.get("/api/curation/exports/{eid}/bundle")
    def bundle(eid: str):
        # Validate the immutable annotation export before packaging source-derived clips.
        download(eid)
        source = corpus.root / "exports" / f"{eid}.jsonl"
        target = corpus.root / "exports" / f"{eid}-{uuid.uuid4().hex}.zip"
        clips = []
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(source, "annotations.jsonl")
            for line in source.read_text().splitlines():
                row = json.loads(line)
                if row["audio"]:
                    _, audio, metadata = evidence(corpus, row["candidate_id"])
                    name = "clips/" + row["candidate_id"] + ".wav"
                    archive.writestr(name, audio)
                    clips.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "file": name,
                            "source_start_s": metadata["clip_start_s"],
                            "source_end_s": metadata["clip_end_s"],
                            "sha256": metadata["clip_sha256"],
                        }
                    )
            archive.writestr("clips.json", json.dumps(clips, indent=2))
        return FileResponse(
            target, media_type="application/zip", filename=f"interruptions-{eid}.zip"
        )

    @app.post("/api/curation/conversations/{cid}/audio")
    async def attach(cid: str, request: Request):
        with corpus.db() as db:
            r = db.execute(
                "SELECT body FROM conversations WHERE id=?", (cid,)
            ).fetchone()
        if not r:
            raise KeyError(cid)
        data = await request.body()
        try:
            info = sf.info(io.BytesIO(data))
        except Exception as exc:
            raise ValueError("Upload a readable WAV file") from exc
        if (
            info.format not in ("WAV", "WAVEX")
            or info.channels not in (1, 2)
            or not 0 < info.duration <= 900
        ):
            raise ValueError("Audio must be mono/stereo WAV, at most 15 minutes")
        last = max((t["end_s"] or 0 for t in json.loads(r["body"])["turns"]), default=0)
        if last > info.duration + 0.05:
            raise ValueError("Conversation timestamps exceed the audio duration")
        name = uuid.uuid4().hex + ".wav"
        path = corpus.root / "assets" / name
        with corpus.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM assets WHERE conversation_id=?", (cid,)
            ).fetchone():
                raise Conflict("Source audio is immutable once attached")
            with path.open("xb") as f:
                f.write(data)
            db.execute(
                "INSERT INTO assets VALUES (?,?,?,?)",
                (cid, name, hashlib.sha256(data).hexdigest(), info.duration),
            )
        return {"attached": True}

    @app.get("/api/curation/conversations/{cid}/audio")
    def audio(cid: str):
        with corpus.db() as db:
            r = db.execute(
                "SELECT * FROM assets WHERE conversation_id=?", (cid,)
            ).fetchone()
        if not r:
            raise KeyError(cid)
        path = corpus.root / "assets" / r["name"]
        if (
            not path.exists()
            or hashlib.sha256(path.read_bytes()).hexdigest() != r["hash"]
        ):
            raise HTTPException(409, "Source audio hash mismatch")
        return FileResponse(path, media_type="audio/wav")

    dist = REPO / "curation-web/dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True))
    return app


def main():
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8766)


if __name__ == "__main__":
    main()
