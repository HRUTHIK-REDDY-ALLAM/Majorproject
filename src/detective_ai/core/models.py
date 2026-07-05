"""Pydantic domain models for the entire Detective AI system."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from detective_ai.core.enums import (
    EvidenceType,
    HypothesisStatus,
    MovementType,
    ObjectionSeverity,
    VerificationStatus,
)

# ── Utility ───────────────────────────────────────────────────────────────────


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ── Evidence Models ───────────────────────────────────────────────────────────


class Evidence(BaseModel):
    """Base evidence item ingested into the system."""

    id: str = Field(default_factory=_uuid)
    type: EvidenceType
    source: str  # e.g. "camera_01", "badge_system", "witness_john"
    timestamp: datetime
    confidence_score: float = Field(ge=0.0, le=1.0, description="Capture quality / reliability")
    metadata: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    created_at: datetime = Field(default_factory=_now)


class VisualDetection(BaseModel):
    """A person or vehicle detected in a video frame."""

    id: str = Field(default_factory=_uuid)
    evidence_id: str  # FK → Evidence
    camera_id: str
    frame_number: int
    timestamp: datetime
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    detection_confidence: float = Field(ge=0.0, le=1.0)
    appearance_embedding: list[float] = Field(default_factory=list)
    label: str = "person"  # person | vehicle
    cluster_id: str | None = None  # assigned identity cluster


class AccessLogEntry(BaseModel):
    """An access-control (badge / card) event."""

    id: str = Field(default_factory=_uuid)
    person_id: str  # badge holder identifier
    person_name: str = ""
    location: str  # door / gate name
    timestamp: datetime
    action: str = "entry"  # entry | exit
    metadata: dict[str, Any] = Field(default_factory=dict)


class WitnessStatement(BaseModel):
    """A free-text witness statement."""

    id: str = Field(default_factory=_uuid)
    source: str  # witness identifier
    text: str
    timestamp: datetime  # when the statement was given
    event_time: datetime | None = None  # when the described event occurred
    reliability_score: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Assessed reliability of the witness"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Hypothesis Models ─────────────────────────────────────────────────────────


class HypothesisEvidence(BaseModel):
    """Link between a hypothesis and a piece of evidence."""

    evidence_id: str
    relationship: str  # "supports" | "contradicts"
    weight: float = Field(default=1.0, ge=0.0, description="Strength of the link")
    reasoning: str = ""


class Hypothesis(BaseModel):
    """A suspect theory under investigation."""

    id: str = Field(default_factory=_uuid)
    case_id: str = ""
    parent_id: str | None = None  # for branching
    title: str
    description: str
    suspect_identity: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    supporting_evidence: list[HypothesisEvidence] = Field(default_factory=list)
    contradicting_evidence: list[HypothesisEvidence] = Field(default_factory=list)
    rejection_reason: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @property
    def evidence_balance(self) -> float:
        """Ratio of supporting vs contradicting evidence weight."""
        sup = sum(e.weight for e in self.supporting_evidence) or 0.001
        con = sum(e.weight for e in self.contradicting_evidence) or 0.001
        return sup / (sup + con)


# ── Trajectory Models ─────────────────────────────────────────────────────────


class TrajectorySegment(BaseModel):
    """A segment of an individual's movement — observed or inferred."""

    id: str = Field(default_factory=_uuid)
    case_id: str = ""
    identity_cluster_id: str  # which person cluster
    from_camera: str
    to_camera: str
    from_time: datetime
    to_time: datetime
    movement_type: MovementType
    confidence: float = Field(ge=0.0, le=1.0)
    possible_routes: list[list[str]] = Field(
        default_factory=list, description="Possible paths through blind spots"
    )
    route_probabilities: list[float] = Field(
        default_factory=list, description="Probability of each route"
    )


# ── Critic / Verification Models ─────────────────────────────────────────────


class CriticObjection(BaseModel):
    """An objection raised by the adversarial critic agent."""

    id: str = Field(default_factory=_uuid)
    hypothesis_id: str
    target_claim: str
    objection_text: str
    severity: ObjectionSeverity
    alternative_explanation: str = ""
    resolved: bool = False
    resolution_text: str = ""
    evidence_cited: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """Result of verifying a single claim in the report."""

    claim_id: str = Field(default_factory=_uuid)
    claim_text: str
    status: VerificationStatus
    evidence_ids: list[str] = Field(default_factory=list)
    notes: str = ""


# ── Report Models ─────────────────────────────────────────────────────────────


class ReportClaim(BaseModel):
    """A single claim in the final investigation report."""

    id: str = Field(default_factory=_uuid)
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    is_confirmed: bool = True  # False if below confidence threshold
    is_inferred: bool = False  # True if based on trajectory inference


class InvestigationReport(BaseModel):
    """The final investigation report."""

    id: str = Field(default_factory=_uuid)
    case_id: str
    title: str
    summary: str
    timeline: list[ReportClaim] = Field(default_factory=list)
    primary_hypothesis: Hypothesis | None = None
    alternative_hypotheses: list[Hypothesis] = Field(
        default_factory=list, description="Considered and rejected alternatives"
    )
    unresolved_objections: list[CriticObjection] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Investigation Session ─────────────────────────────────────────────────────


class InvestigationCase(BaseModel):
    """Top-level investigation case."""

    id: str = Field(default_factory=_uuid)
    title: str
    description: str = ""
    status: str = "pending"  # pending | running | completed | failed
    phase: str = "ingestion"
    evidence_ids: list[str] = Field(default_factory=list)
    hypothesis_ids: list[str] = Field(default_factory=list)
    current_round: int = 0
    max_rounds: int = 3
    created_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None
    report_id: str | None = None
