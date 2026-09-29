"""Loopback-only application API with same-origin write authorization."""

import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .contracts import StartRun
from .service import Coordinator

REPO = Path(__file__).resolve().parents[2]


def create_app(coordinator=None):
    manager = coordinator or Coordinator(REPO)
    token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()

    app = FastAPI(title="Repair Bench", lifespan=lifespan)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"]
    )

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                return JSONResponse(
                    {"detail": "Foreign origin rejected"}, status_code=403
                )
            if not secrets.compare_digest(
                request.headers.get("x-repair-token", ""), token
            ):
                return JSONResponse(
                    {"detail": "Missing session token"}, status_code=403
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "Run not found"}, status_code=404)

    @app.get("/api/v1/bootstrap")
    def bootstrap():
        return {
            "token": token,
            "active_run": manager.active,
            "capabilities": {
                "controlled_completed_utterances": True,
                "live": False,
                "C1": False,
                "microphone": False,
                "interruption_metrics": False,
            },
        }

    @app.get("/api/v1/runs")
    def runs():
        return manager.store.list()

    @app.post("/api/v1/runs", status_code=202)
    def start(body: StartRun):
        try:
            return {"id": manager.start(body.scenario_id)}
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/runs/{run}")
    def detail(run: str):
        return manager.store.detail(run)

    @app.get("/api/v1/runs/{run}/events")
    def events(run: str):
        return manager.store.events(run)

    @app.get("/api/v1/runs/{run}/verify")
    def verify(run: str):
        try:
            return manager.store.verify(run)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/v1/runs/{run}/audio/{asset}")
    def audio(run: str, asset: str):
        folder = manager.store.folder(run)
        if asset not in {
            f"{kind}-{i}.wav" for kind in ("input", "response") for i in range(2)
        }:
            raise HTTPException(404)
        path = folder / asset
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/v1/runs/{run}/manifest")
    def manifest(run: str):
        path = manager.store.folder(run) / "manifest.json"
        if not path.exists():
            raise HTTPException(409, "Run has not finalized")
        return FileResponse(
            path, media_type="application/json", filename=f"repair-bench-{run}.json"
        )

    dist = REPO / "web/dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


def main():
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
