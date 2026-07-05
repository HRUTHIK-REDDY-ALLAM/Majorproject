"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # ── Groq LLM ──────────────────────────────────────────────
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    groq_rate_limit_rpm: int = field(
        default_factory=lambda: int(os.getenv("GROQ_RATE_LIMIT_RPM", "28"))
    )

    # ── PostgreSQL + pgvector ─────────────────────────────────
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://detective:detective123@localhost:5432/detective_ai",
        )
    )

    # ── Application ───────────────────────────────────────────
    app_host: str = field(default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0"))
    app_port: int = field(default_factory=lambda: int(os.getenv("APP_PORT", "8000")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # ── CV Pipeline ───────────────────────────────────────────
    yolo_model: str = field(
        default_factory=lambda: os.getenv("YOLO_MODEL", "yolov8n.pt")
    )
    yolo_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.5"))
    )
    reid_batch_size: int = field(
        default_factory=lambda: int(os.getenv("REID_BATCH_SIZE", "4"))
    )
    frame_extraction_fps: int = field(
        default_factory=lambda: int(os.getenv("FRAME_EXTRACTION_FPS", "2"))
    )

    # ── Agent Settings ────────────────────────────────────────
    max_investigation_rounds: int = field(
        default_factory=lambda: int(os.getenv("MAX_INVESTIGATION_ROUNDS", "3"))
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
    )

    # ── Embeddings ────────────────────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "384"))
    )

    # ── Paths ─────────────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    @property
    def cv_models_dir(self) -> Path:
        return self.project_root / "src" / "detective_ai" / "cv" / "models"

    @property
    def benchmarks_dir(self) -> Path:
        return self.project_root / "benchmarks" / "scenarios"


# Singleton instance
settings = Settings()
