"""API request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ── Request Schemas ───────────────────────────────────────────


class IngestVideoRequest(BaseModel):
    camera_id: str
    start_time: datetime
    case_id: str = ""
    fps: int = 2
    max_frames: int | None = None


class IngestLogsRequest(BaseModel):
    content: str
    format: str = "json"  # "json" or "csv"
    source: str = "badge_system"
    case_id: str = ""


class IngestStatementRequest(BaseModel):
    text: str
    source: str
    timestamp: datetime
    event_time: datetime | None = None
    reliability_score: float = 0.7
    case_id: str = ""


class InvestigateRequest(BaseModel):
    case_id: str
    title: str
    description: str = ""
    max_rounds: int = 3


class CounterfactualRequest(BaseModel):
    case_id: str
    removed_evidence_id: str


# ── Response Schemas ──────────────────────────────────────────


class StatusResponse(BaseModel):
    status: str
    message: str
    data: dict[str, Any] | None = None


class CaseResponse(BaseModel):
    id: str
    title: str
    status: str
    phase: str
    current_round: int
    created_at: datetime | None = None


class EvidenceResponse(BaseModel):
    id: str
    type: str
    source: str
    timestamp: datetime | None = None
    confidence_score: float
    description: str


class ReportResponse(BaseModel):
    case_id: str
    report: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
