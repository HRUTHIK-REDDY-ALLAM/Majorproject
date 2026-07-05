"""Enumeration types used across the system."""

from __future__ import annotations

from enum import Enum


class EvidenceType(str, Enum):
    """Types of evidence the system can ingest."""

    VIDEO_FRAME = "video_frame"
    ACCESS_LOG = "access_log"
    WITNESS_STATEMENT = "witness_statement"
    VISUAL_DETECTION = "visual_detection"
    INFERRED_TRAJECTORY = "inferred_trajectory"


class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis."""

    ACTIVE = "active"
    PRUNED = "pruned"
    CONFIRMED = "confirmed"
    UNDER_REVIEW = "under_review"


class ObjectionSeverity(str, Enum):
    """Severity levels for critic objections."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class VerificationStatus(str, Enum):
    """Result of verifying a single claim."""

    PASSED = "passed"
    FAILED = "failed"
    UNVERIFIABLE = "unverifiable"


class MovementType(str, Enum):
    """Whether a trajectory segment is directly observed or inferred."""

    OBSERVED = "observed"
    INFERRED = "inferred"


class InvestigationPhase(str, Enum):
    """Current phase of the investigation pipeline."""

    INGESTION = "ingestion"
    DETECTION = "detection"
    HYPOTHESIS_FORMATION = "hypothesis_formation"
    EVIDENCE_GATHERING = "evidence_gathering"
    GAP_INFERENCE = "gap_inference"
    ADVERSARIAL_REVIEW = "adversarial_review"
    VERIFICATION = "verification"
    REPORT_GENERATION = "report_generation"
    COMPLETED = "completed"
    FAILED = "failed"
