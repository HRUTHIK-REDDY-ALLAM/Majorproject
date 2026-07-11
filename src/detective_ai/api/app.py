"""FastAPI application: main entry point for Detective AI."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detective_ai.api.routes.ingest import router as ingest_router
from detective_ai.api.routes.investigate import router as investigate_router
from detective_ai.api.routes.report import counterfactual_router, report_router
from detective_ai.config import settings
from detective_ai.storage.database import db

logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────

app = FastAPI(
    title="Detective AI",
    description="Multi-Agent Forensic Reasoning System",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────

app.include_router(ingest_router)
app.include_router(investigate_router)
app.include_router(report_router)
app.include_router(counterfactual_router)


# ── Static Files (Frontend) ──────────────────────────────────

def _resolve_frontend_dir() -> Path:
    """Locate the frontend directory whether running from a source checkout
    (frontend/ four levels up from this file) or from an installed package
    in a container with WORKDIR/frontend copied alongside (cwd-relative)."""
    candidates = [
        Path.cwd() / "frontend",
        Path(__file__).resolve().parent.parent.parent.parent / "frontend",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


_FRONTEND_DIR = _resolve_frontend_dir()

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR / "static")), name="static")


# ── Health & Info ─────────────────────────────────────────────


@app.get("/api/health")
async def health():
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/api/v1/cases")
async def list_cases():
    """List all investigation cases."""
    with db.session() as session:
        cases = db.list_cases(session)
        return {
            "cases": [
                {
                    "id": c.id,
                    "title": c.title,
                    "status": c.status,
                    "phase": c.phase,
                    "current_round": c.current_round,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "completed_at": c.completed_at.isoformat() if c.completed_at else None,
                }
                for c in cases
            ]
        }


# ── Serve Frontend ────────────────────────────────────────────


@app.get("/")
async def serve_frontend():
    """Serve the frontend application."""
    index_path = _FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Detective AI API is running. Visit /api/docs for API documentation."}


# ── Startup/Shutdown ──────────────────────────────────────────


@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    try:
        db.init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning("API running without database. Some features may not work.")


# ── CLI Entry Point ───────────────────────────────────────────


def main():
    """Run the server from command line."""
    import uvicorn

    logging.basicConfig(level=getattr(logging, settings.log_level))
    uvicorn.run(
        "detective_ai.api.app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
