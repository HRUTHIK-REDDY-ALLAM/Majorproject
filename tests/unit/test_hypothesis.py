"""Unit tests for the hypothesis tracker."""


from detective_ai.core.models import Hypothesis
from detective_ai.hypothesis.tracker import HypothesisTracker


class TestHypothesisTracker:
    def setup_method(self):
        self.tracker = HypothesisTracker(prune_threshold=0.15, max_active=3)

    def test_add_hypothesis(self):
        h = Hypothesis(
            title="Test hypothesis",
            description="Test",
            confidence=0.5,
        )
        self.tracker.add_hypothesis(h)
        assert self.tracker.get_hypothesis(h.id) is not None

    def test_leading_hypothesis(self):
        h1 = Hypothesis(title="Low conf", description="", confidence=0.3)
        h2 = Hypothesis(title="High conf", description="", confidence=0.8)
        self.tracker.add_hypothesis(h1)
        self.tracker.add_hypothesis(h2)

        leading = self.tracker.get_leading_hypothesis()
        assert leading is not None
        assert leading.id == h2.id

    def test_prune_low_confidence(self):
        h = Hypothesis(title="Weak theory", description="", confidence=0.1)
        self.tracker.add_hypothesis(h)

        # Should be auto-pruned
        assert self.tracker.get_hypothesis(h.id) is None
        pruned = self.tracker.get_pruned_hypotheses()
        assert len(pruned) == 1
        assert pruned[0].rejection_reason is not None

    def test_branch_hypothesis(self):
        parent = Hypothesis(
            title="Parent theory",
            description="Base hypothesis",
            confidence=0.6,
        )
        self.tracker.add_hypothesis(parent)

        child_id = self.tracker.branch_hypothesis(
            parent_id=parent.id,
            title="Child theory",
            description="Branched variant",
        )

        child = self.tracker.get_hypothesis(child_id)
        assert child is not None
        assert child.parent_id == parent.id

    def test_max_active_pruning(self):
        # Add more than max_active hypotheses
        for i in range(5):
            h = Hypothesis(
                title=f"Theory {i}",
                description="",
                confidence=0.3 + i * 0.1,
            )
            self.tracker.add_hypothesis(h)

        active = self.tracker.get_active_hypotheses()
        assert len(active) <= 3  # max_active

    def test_add_evidence_updates_confidence(self):
        h = Hypothesis(title="Test", description="", confidence=0.5)
        self.tracker.add_hypothesis(h)

        self.tracker.add_supporting_evidence(h.id, "evidence_1", weight=2.0)
        updated = self.tracker.get_hypothesis(h.id)
        # Confidence should increase with supporting evidence
        assert updated is not None


class TestConfidenceEngine:
    def test_ece_computation(self):
        from detective_ai.hypothesis.confidence import ConfidenceEngine

        # Perfect calibration: 80% confident, 80% correct
        confs = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
        outcomes = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # 80% correct

        ece = ConfidenceEngine.expected_calibration_error(confs, outcomes)
        assert ece < 0.1  # Should be well-calibrated

    def test_reliability_diagram(self):
        from detective_ai.hypothesis.confidence import ConfidenceEngine

        confs = [0.1, 0.3, 0.5, 0.7, 0.9] * 4
        outcomes = [0, 0, 1, 1, 1] * 4

        diagram = ConfidenceEngine.reliability_diagram(confs, outcomes, n_bins=5)
        assert "bin_centers" in diagram
        assert "accuracies" in diagram
        assert len(diagram["bin_centers"]) == 5
