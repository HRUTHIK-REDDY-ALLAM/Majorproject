"""Unit tests for core data models."""

from datetime import datetime

import pytest

from detective_ai.core.enums import (
    EvidenceType,
    HypothesisStatus,
    MovementType,
    ObjectionSeverity,
)
from detective_ai.core.models import (
    CriticObjection,
    Evidence,
    Hypothesis,
    HypothesisEvidence,
    TrajectorySegment,
)


class TestEvidence:
    def test_create_evidence(self):
        e = Evidence(
            type=EvidenceType.ACCESS_LOG,
            source="badge_system",
            timestamp=datetime(2025, 1, 15, 8, 30),
            confidence_score=0.95,
            description="Entry at main door",
        )
        assert e.id is not None
        assert e.type == EvidenceType.ACCESS_LOG
        assert e.confidence_score == 0.95

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            Evidence(
                type=EvidenceType.VIDEO_FRAME,
                source="cam_01",
                timestamp=datetime.now(),
                confidence_score=1.5,  # out of bounds
            )


class TestHypothesis:
    def test_create_hypothesis(self):
        h = Hypothesis(
            title="Suspect A entered server room",
            description="Based on badge log evidence",
            confidence=0.7,
        )
        assert h.status == HypothesisStatus.ACTIVE
        assert h.confidence == 0.7

    def test_evidence_balance(self):
        h = Hypothesis(
            title="Test hypothesis",
            description="Testing evidence balance",
            supporting_evidence=[
                HypothesisEvidence(evidence_id="e1", relationship="supports", weight=2.0),
                HypothesisEvidence(evidence_id="e2", relationship="supports", weight=1.0),
            ],
            contradicting_evidence=[
                HypothesisEvidence(evidence_id="e3", relationship="contradicts", weight=1.0),
            ],
        )
        # sup = 3.0, con = 1.0, balance = 3.0 / (3.0 + 1.0) = 0.75
        assert h.evidence_balance == 0.75


class TestTrajectorySegment:
    def test_inferred_segment(self):
        seg = TrajectorySegment(
            identity_cluster_id="cluster_0",
            from_camera="cam_01",
            to_camera="cam_03",
            from_time=datetime(2025, 1, 15, 9, 0),
            to_time=datetime(2025, 1, 15, 9, 5),
            movement_type=MovementType.INFERRED,
            confidence=0.6,
            possible_routes=[["cam_01", "cam_02", "cam_03"]],
            route_probabilities=[0.8],
        )
        assert seg.movement_type == MovementType.INFERRED
        assert len(seg.possible_routes) == 1


class TestCriticObjection:
    def test_create_objection(self):
        obj = CriticObjection(
            hypothesis_id="hyp_1",
            target_claim="Suspect was in server room at 9 PM",
            objection_text="Badge log shows suspect exited building at 8:45 PM",
            severity=ObjectionSeverity.CRITICAL,
            alternative_explanation="Tailgating or borrowed badge",
        )
        assert obj.severity == ObjectionSeverity.CRITICAL
        assert obj.resolved is False
