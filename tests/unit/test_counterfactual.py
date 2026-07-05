"""Tests for the counterfactual exploration engine."""

from __future__ import annotations

from detective_ai.core.models import Hypothesis
from detective_ai.hypothesis.counterfactual import CounterfactualEngine
from detective_ai.hypothesis.tracker import HypothesisTracker


def _build_tracker() -> tuple[HypothesisTracker, str, str]:
    tracker = HypothesisTracker(prune_threshold=0.05)
    h1 = Hypothesis(title="Suspect A did it", description="", confidence=0.5)
    h2 = Hypothesis(title="Suspect B did it", description="", confidence=0.5)
    tracker.add_hypothesis(h1)
    tracker.add_hypothesis(h2)
    # h1 is supported by the key evidence; h2 by weaker independent evidence
    tracker.add_supporting_evidence(h1.id, "ev_key", weight=3.0, reasoning="camera")
    tracker.add_supporting_evidence(h1.id, "ev_minor", weight=0.5, reasoning="badge")
    tracker.add_supporting_evidence(h2.id, "ev_other", weight=1.0, reasoning="witness")
    return tracker, h1.id, h2.id


class TestCounterfactualEngine:
    def test_removing_key_evidence_changes_confidence(self):
        tracker, h1_id, _ = _build_tracker()
        engine = CounterfactualEngine(tracker)
        result = engine.explore("ev_key")

        changed = {c["hypothesis_id"]: c for c in result["changes"]}
        assert h1_id in changed
        assert changed[h1_id]["delta"] < 0

    def test_original_tracker_untouched(self):
        tracker, h1_id, _ = _build_tracker()
        before = tracker.get_hypothesis(h1_id).confidence
        CounterfactualEngine(tracker).explore("ev_key")
        assert tracker.get_hypothesis(h1_id).confidence == before
        assert len(tracker.get_hypothesis(h1_id).supporting_evidence) == 2

    def test_removing_irrelevant_evidence_changes_nothing(self):
        tracker, _, _ = _build_tracker()
        result = CounterfactualEngine(tracker).explore("ev_nonexistent")
        assert result["conclusion_changed"] is False
        assert result["changes"] == []

    def test_reports_leading_hypothesis_swap(self):
        tracker, h1_id, h2_id = _build_tracker()
        assert tracker.get_leading_hypothesis().id == h1_id
        result = CounterfactualEngine(tracker).explore("ev_key")
        # Without its key evidence, h1 (0.5 vs 1.0 weight) drops below h2
        assert result["original_leading"]["id"] == h1_id
        assert result["counterfactual_leading"]["id"] == h2_id
        assert result["conclusion_changed"] is True
