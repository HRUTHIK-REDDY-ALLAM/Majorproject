"""Tests for the benchmark generator, evaluator, and confidence calibration."""

from __future__ import annotations

import pytest

from benchmarks.evaluator import BenchmarkEvaluator
from benchmarks.generator import generate_all_scenarios, generate_scenario
from detective_ai.hypothesis.confidence import ConfidenceEngine


class TestGenerator:
    def test_scenario_structure(self):
        sc = generate_scenario(0, seed=1)
        assert sc["ground_truth"]["suspect"]["id"]
        assert sc["evidence"]["camera_sightings"]
        assert sc["evidence"]["access_logs"]
        assert sc["evidence"]["witness_statements"]
        assert sc["false_leads"], "every scenario must plant a false lead"
        assert sc["metadata"]["num_blind_spots"] >= 1

    def test_deterministic_with_seed(self):
        a = generate_scenario(3, seed=99)
        b = generate_scenario(3, seed=99)
        assert a["ground_truth"]["suspect"] == b["ground_truth"]["suspect"]
        assert a["evidence"]["access_logs"] == b["evidence"]["access_logs"]

    def test_suite_generation(self, tmp_path):
        scenarios = generate_all_scenarios(count=3, output_dir=tmp_path)
        assert len(scenarios) == 3
        assert (tmp_path / "scenario_00" / "ground_truth.json").exists()


class TestEvaluator:
    def _report_for(self, scenario, correct=True, confidence=0.8):
        suspect_id = scenario["ground_truth"]["suspect"]["id"]
        predicted = suspect_id if correct else "EMP999"
        cameras = {p["camera_id"] for p in scenario["ground_truth"]["suspect_path"]}
        return {
            "primary_conclusion": {"hypothesis": f"{predicted} did it"},
            "timeline": [{"camera_id": c} for c in cameras],
            "confidence_assessment": {"overall_confidence": confidence},
            "unresolved_objections": [],
            "alternative_hypotheses": [
                {"title": scenario["false_leads"][0]["person_implicated"]["name"]}
            ],
            "metadata": {"hypotheses_considered": 5, "hypotheses_rejected": 3},
        }

    def test_correct_identification_scored(self):
        sc = generate_scenario(0, seed=5)
        ev = BenchmarkEvaluator()
        result = ev.evaluate_scenario(sc, self._report_for(sc, correct=True))
        assert result["suspect_correct"] is True
        assert result["timeline_accuracy"] == 1.0
        assert result["critic_effectiveness"] == 1.0

    def test_wrong_identification_scored(self):
        sc = generate_scenario(1, seed=6)
        ev = BenchmarkEvaluator()
        result = ev.evaluate_scenario(sc, self._report_for(sc, correct=False))
        assert result["suspect_correct"] is False
        assert result["false_positive"] is True

    def test_aggregate_metrics(self):
        ev = BenchmarkEvaluator()
        for i in range(4):
            sc = generate_scenario(i, seed=10 + i)
            ev.evaluate_scenario(sc, self._report_for(sc, correct=(i % 2 == 0)))
        agg = ev.compute_aggregate_metrics()
        assert agg["total_scenarios"] == 4
        assert agg["suspect_identification"]["accuracy"] == 0.5
        assert 0.0 <= agg["confidence_calibration"]["expected_calibration_error"] <= 1.0


class TestConfidenceEngine:
    def test_no_evidence_means_max_uncertainty(self):
        engine = ConfidenceEngine()
        assert engine.compute_confidence([], [], [], []) == 0.5

    def test_supporting_evidence_raises_confidence(self):
        engine = ConfidenceEngine()
        conf = engine.compute_confidence(
            [2.0, 1.5], [], [0.9, 0.9], [0.9, 0.9],
        )
        assert conf > 0.5

    def test_contradicting_evidence_lowers_confidence(self):
        engine = ConfidenceEngine()
        conf = engine.compute_confidence(
            [0.5], [2.0, 2.0], [0.9, 0.9, 0.9], [0.9, 0.9, 0.9],
        )
        assert conf < 0.5

    def test_perfect_calibration_has_zero_ece(self):
        # 10 predictions at 0.75 confidence with 75% accuracy is well calibrated
        confs = [0.75] * 8
        outcomes = [1, 1, 1, 1, 1, 1, 0, 0]
        ece = ConfidenceEngine.expected_calibration_error(confs, outcomes)
        assert ece == pytest.approx(0.0, abs=1e-9)

    def test_overconfidence_detected(self):
        confs = [0.95] * 10
        outcomes = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ece = ConfidenceEngine.expected_calibration_error(confs, outcomes)
        assert ece == pytest.approx(0.85, abs=0.01)

    def test_platt_calibration_reduces_ece(self):
        engine = ConfidenceEngine()
        # Systematically overconfident predictor
        preds = [0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.8]
        outcomes = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
        fit = engine.calibrate(preds, outcomes)
        assert "platt_a" in fit
        calibrated = engine.compute_confidence(
            [2.0, 2.0, 2.0], [], [0.9] * 3, [0.9] * 3,
        )
        assert 0.0 <= calibrated <= 1.0

    def test_reliability_diagram_shape(self):
        diagram = ConfidenceEngine.reliability_diagram(
            [0.1, 0.5, 0.9], [0, 1, 1], n_bins=10,
        )
        assert len(diagram["bin_centers"]) == 10
        assert sum(diagram["counts"]) == 3
