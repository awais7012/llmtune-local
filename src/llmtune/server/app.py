"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llmtune.server.routers import (
    config_router,
    dataset_router,
    jobs_router,
    kaggle_router,
    library_router,
    models_router,
)


def create_app() -> FastAPI:
    app = FastAPI(title="llmtune", version="0.1.0")

    # Local-only tool: allow the bundled UI (served from any localhost port)
    # but not arbitrary websites, so a random browser tab can't reach the API.
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(config_router)
    app.include_router(dataset_router)
    app.include_router(models_router)
    app.include_router(jobs_router)
    app.include_router(kaggle_router)
    app.include_router(library_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    from llmtune.server.config import static_dir

    static = static_dir()
    if not static:
        return

    assets = static / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith(("api/", "auth/", "health")):
            return {"detail": "Not found"}
        file = static / full_path
        if file.is_file():
            return FileResponse(
                file,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return FileResponse(
            static / "index.html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
