"""Database adapter: PostgreSQL + pgvector in production, SQLite fallback locally.

Manages connection pooling, schema creation, and CRUD for all domain tables.
Uses SQLAlchemy 2.0. On PostgreSQL, embeddings use the pgvector extension and
JSON columns use JSONB; on SQLite (local dev / CI), embeddings are stored as
JSON text and similarity search falls back to in-Python cosine similarity.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    TypeDecorator,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from detective_ai.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


# ── Portable column types ─────────────────────────────────────────────────────

# JSONB on PostgreSQL, generic JSON elsewhere
PortableJSON = JSON().with_variant(JSONB(), "postgresql")


class EmbeddingVector(TypeDecorator):
    """pgvector Vector on PostgreSQL; JSON-encoded TEXT on other dialects."""

    impl = Text
    cache_ok = True

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value, dialect):
        if value is None or dialect.name == "postgresql":
            return value
        return json.dumps([float(x) for x in value])

    def process_result_value(self, value, dialect):
        if value is None or dialect.name == "postgresql":
            return value
        return json.loads(value)


# ── ORM Models ────────────────────────────────────────────────────────────────


class EvidenceRow(Base):
    __tablename__ = "evidence"

    id = Column(String, primary_key=True)
    type = Column(String(50), nullable=False)
    source = Column(String(200), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    confidence_score = Column(Float, default=0.5)
    description = Column(Text, default="")
    metadata_ = Column("metadata", PortableJSON, default={})
    embedding = Column(EmbeddingVector(384), nullable=True)  # sentence-transformer dim
    created_at = Column(DateTime, default=datetime.utcnow)


class VisualDetectionRow(Base):
    __tablename__ = "visual_detections"

    id = Column(String, primary_key=True)
    evidence_id = Column(String, ForeignKey("evidence.id"), nullable=False)
    camera_id = Column(String(50), nullable=False)
    frame_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_w = Column(Float)
    bbox_h = Column(Float)
    detection_confidence = Column(Float, default=0.5)
    label = Column(String(50), default="person")
    cluster_id = Column(String, nullable=True)
    appearance_embedding = Column(EmbeddingVector(512), nullable=True)  # ReID dim


class HypothesisRow(Base):
    __tablename__ = "hypotheses"

    id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False)
    parent_id = Column(String, ForeignKey("hypotheses.id"), nullable=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, default="")
    suspect_identity = Column(String, nullable=True)
    confidence = Column(Float, default=0.5)
    status = Column(String(20), default="active")
    rejection_reason = Column(Text, nullable=True)
    supporting_evidence = Column(PortableJSON, default=[])
    contradicting_evidence = Column(PortableJSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


class TrajectoryRow(Base):
    __tablename__ = "trajectories"

    id = Column(String, primary_key=True)
    case_id = Column(String, nullable=False)
    identity_cluster_id = Column(String, nullable=False)
    from_camera = Column(String(50), nullable=False)
    to_camera = Column(String(50), nullable=False)
    from_time = Column(DateTime, nullable=False)
    to_time = Column(DateTime, nullable=False)
    movement_type = Column(String(20), default="observed")
    confidence = Column(Float, default=0.5)
    possible_routes = Column(PortableJSON, default=[])
    route_probabilities = Column(PortableJSON, default=[])


class AccessLogRow(Base):
    __tablename__ = "access_logs"

    id = Column(String, primary_key=True)
    person_id = Column(String, nullable=False)
    person_name = Column(String(200), default="")
    location = Column(String(200), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    action = Column(String(20), default="entry")
    metadata_ = Column("metadata", PortableJSON, default={})


class WitnessStatementRow(Base):
    __tablename__ = "witness_statements"

    id = Column(String, primary_key=True)
    source = Column(String(200), nullable=False)
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    event_time = Column(DateTime, nullable=True)
    reliability_score = Column(Float, default=0.7)
    metadata_ = Column("metadata", PortableJSON, default={})
    embedding = Column(EmbeddingVector(384), nullable=True)


class InvestigationCaseRow(Base):
    __tablename__ = "investigation_cases"

    id = Column(String, primary_key=True)
    title = Column(String(500), nullable=False)
    description = Column(Text, default="")
    status = Column(String(20), default="pending")
    phase = Column(String(50), default="ingestion")
    current_round = Column(Integer, default=0)
    max_rounds = Column(Integer, default=3)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    report_data = Column(PortableJSON, nullable=True)


# ── Database Manager ──────────────────────────────────────────────────────────


def _make_engine(url: str):
    """Create an engine with dialect-appropriate options."""
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_size=5, max_overflow=10, pool_pre_ping=True)


class DatabaseManager:
    """Manages the database connection and operations.

    Uses the configured DATABASE_URL (PostgreSQL in production). If that
    server is unreachable at init_db() time, falls back to a local SQLite
    file so the system remains fully functional for development and CI.
    """

    def __init__(self, database_url: str | None = None) -> None:
        self._url = database_url or settings.database_url
        self._engine = _make_engine(self._url)
        self._session_factory = sessionmaker(bind=self._engine)

    @property
    def url(self) -> str:
        return self._url

    @property
    def is_postgres(self) -> bool:
        return self._engine.dialect.name == "postgresql"

    def _switch_to_sqlite_fallback(self) -> None:
        data_dir = settings.project_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._url = f"sqlite:///{(data_dir / 'detective_ai.db').as_posix()}"
        self._engine = _make_engine(self._url)
        self._session_factory = sessionmaker(bind=self._engine)
        logger.warning(f"PostgreSQL unreachable — falling back to SQLite at {self._url}")

    def init_db(self) -> None:
        """Create all tables (and the pgvector extension on PostgreSQL)."""
        try:
            with self._engine.connect() as conn:
                if self.is_postgres:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
        except OperationalError as e:
            if self.is_postgres:
                logger.warning(f"Could not connect to PostgreSQL: {e}")
                self._switch_to_sqlite_fallback()
            else:
                raise
        Base.metadata.create_all(self._engine)
        logger.info(f"Database initialized ({self._engine.dialect.name}).")

    def drop_all(self) -> None:
        """Drop all tables. Use with caution."""
        Base.metadata.drop_all(self._engine)
        logger.warning("All database tables dropped.")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for a database session."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Evidence CRUD ─────────────────────────────────────────

    def insert_evidence(self, session: Session, **kwargs: Any) -> EvidenceRow:
        row = EvidenceRow(**kwargs)
        session.add(row)
        session.flush()
        return row

    def get_evidence(self, session: Session, evidence_id: str) -> EvidenceRow | None:
        return session.query(EvidenceRow).filter_by(id=evidence_id).first()

    def get_evidence_by_case(
        self, session: Session, case_id: str, evidence_type: str | None = None
    ) -> list[EvidenceRow]:
        q = session.query(EvidenceRow)
        if evidence_type:
            q = q.filter_by(type=evidence_type)
        return q.all()

    # ── Visual Detection CRUD ─────────────────────────────────

    def insert_detection(self, session: Session, **kwargs: Any) -> VisualDetectionRow:
        row = VisualDetectionRow(**kwargs)
        session.add(row)
        session.flush()
        return row

    def get_detections_by_camera(
        self, session: Session, camera_id: str, start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[VisualDetectionRow]:
        q = session.query(VisualDetectionRow).filter_by(camera_id=camera_id)
        if start_time:
            q = q.filter(VisualDetectionRow.timestamp >= start_time)
        if end_time:
            q = q.filter(VisualDetectionRow.timestamp <= end_time)
        return q.order_by(VisualDetectionRow.timestamp).all()

    # ── Hypothesis CRUD ───────────────────────────────────────

    def insert_hypothesis(self, session: Session, **kwargs: Any) -> HypothesisRow:
        row = HypothesisRow(**kwargs)
        session.add(row)
        session.flush()
        return row

    def get_hypotheses(self, session: Session, case_id: str) -> list[HypothesisRow]:
        return session.query(HypothesisRow).filter_by(case_id=case_id).all()

    def update_hypothesis(
        self, session: Session, hypothesis_id: str, **kwargs: Any
    ) -> HypothesisRow | None:
        row = session.query(HypothesisRow).filter_by(id=hypothesis_id).first()
        if row:
            for k, v in kwargs.items():
                setattr(row, k, v)
            row.updated_at = datetime.utcnow()
            session.flush()
        return row

    # ── Investigation Case CRUD ───────────────────────────────

    def insert_case(self, session: Session, **kwargs: Any) -> InvestigationCaseRow:
        row = InvestigationCaseRow(**kwargs)
        session.add(row)
        session.flush()
        return row

    def get_case(self, session: Session, case_id: str) -> InvestigationCaseRow | None:
        return session.query(InvestigationCaseRow).filter_by(id=case_id).first()

    def update_case(
        self, session: Session, case_id: str, **kwargs: Any
    ) -> InvestigationCaseRow | None:
        row = session.query(InvestigationCaseRow).filter_by(id=case_id).first()
        if row:
            for k, v in kwargs.items():
                setattr(row, k, v)
            session.flush()
        return row

    def list_cases(self, session: Session) -> list[InvestigationCaseRow]:
        return session.query(InvestigationCaseRow).order_by(
            InvestigationCaseRow.created_at.desc()
        ).all()

    # ── Access Logs ───────────────────────────────────────────

    def insert_access_log(self, session: Session, **kwargs: Any) -> AccessLogRow:
        row = AccessLogRow(**kwargs)
        session.add(row)
        session.flush()
        return row

    def get_access_logs(
        self, session: Session, person_id: str | None = None,
        location: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[AccessLogRow]:
        q = session.query(AccessLogRow)
        if person_id:
            q = q.filter_by(person_id=person_id)
        if location:
            q = q.filter_by(location=location)
        if start_time:
            q = q.filter(AccessLogRow.timestamp >= start_time)
        if end_time:
            q = q.filter(AccessLogRow.timestamp <= end_time)
        return q.order_by(AccessLogRow.timestamp).all()

    # ── Witness Statements ────────────────────────────────────

    def insert_statement(self, session: Session, **kwargs: Any) -> WitnessStatementRow:
        row = WitnessStatementRow(**kwargs)
        session.add(row)
        session.flush()
        return row

    # ── Trajectories ──────────────────────────────────────────

    def insert_trajectory(self, session: Session, **kwargs: Any) -> TrajectoryRow:
        row = TrajectoryRow(**kwargs)
        session.add(row)
        session.flush()
        return row

    def get_trajectories(
        self, session: Session, case_id: str, cluster_id: str | None = None
    ) -> list[TrajectoryRow]:
        q = session.query(TrajectoryRow).filter_by(case_id=case_id)
        if cluster_id:
            q = q.filter_by(identity_cluster_id=cluster_id)
        return q.order_by(TrajectoryRow.from_time).all()


# ── Module-level singleton ────────────────────────────────────────────────────

db = DatabaseManager()
