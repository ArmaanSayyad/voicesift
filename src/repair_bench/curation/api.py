import hashlib, io, json, secrets, uuid, zipfile
from pathlib import Path
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .store import Corpus, Conflict
from .models import Import, Review, Analyze
from .analysis import Analysis, evidence
from fastapi.responses import Response

REPO = Path(__file__).resolve().parents[3]
MAX_AUDIO = 50_000_000


def create_app(root=None, backend=None):
    corpus = Corpus(root or REPO / "artifacts/interruption-curation")
    analysis = Analysis(corpus, backend)
    token = secrets.token_urlsafe(32)
    app = FastAPI(title="Interruption Curation")
    app.state.corpus = corpus
    app.state.analysis = analysis
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
            limit = MAX_AUDIO if request.url.path.endswith("/audio") else 2_100_000
            total = 0
            chunks = []
            async for chunk in request.stream():
                total += len(chunk)
                if total > limit:
                    return JSONResponse({"detail": "Upload too large"}, status_code=413)
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
        }

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
