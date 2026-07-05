"""End-to-end tests for the deterministic offline investigation pipeline."""

from __future__ import annotations

import json

import pytest

from benchmarks.generator import generate_scenario
from detective_ai.pipeline.offline import OfflineInvestigationPipeline


@pytest.fixture(scope="module")
def scenario():
    return generate_scenario(0, seed=42)


@pytest.fixture(scope="module")
def report(scenario):
    pipeline = OfflineInvestigationPipeline(scenario["topology"])
    return pipeline.investigate(
        case_id=scenario["id"],
        case_title=scenario["title"],
        case_description=scenario["description"],
        camera_sightings=scenario["evidence"]["camera_sightings"],
        access_logs=scenario["evidence"]["access_logs"],
        witness_statements=scenario["evidence"]["witness_statements"],
    )


class TestOfflinePipeline:
    def test_identifies_correct_suspect(self, scenario, report):
        assert (
            report["primary_conclusion"]["suspect_id"]
            == scenario["ground_truth"]["suspect"]["id"]
        )

    def test_confidence_is_bounded_and_calibratable(self, report):
        conf = report["confidence_assessment"]["overall_confidence"]
        assert 0.0 < conf < 1.0

    def test_every_timeline_claim_cites_evidence(self, report):
        assert report["timeline"], "timeline must not be empty"
        for entry in report["timeline"]:
            assert entry["evidence_ids"], f"uncited claim: {entry['event']}"

    def test_inferred_movement_flagged_distinctly(self, report):
        inferred = [t for t in report["timeline"] if t["movement_type"] == "inferred"]
        observed = [t for t in report["timeline"] if t["movement_type"] == "observed"]
        assert observed, "expected observed segments"
        # The generator plants at least one blind spot in the suspect path
        assert inferred, "expected inferred blind-spot segments"
        for seg in inferred:
            assert seg.get("possible_routes") is not None

    def test_false_lead_flagged_by_critic(self, scenario, report):
        decoy_name = scenario["false_leads"][0]["person_implicated"]["name"]
        all_objections = json.dumps(
            report["unresolved_objections"] + report["resolved_objections"]
        )
        assert decoy_name in all_objections

    def test_rejected_alternatives_logged_with_reasons(self, report):
        rejected = [
            h for h in report["alternative_hypotheses"] if h["status"] == "pruned"
        ]
        assert rejected
        for h in rejected:
            assert h["rejection_reason"]

    def test_verification_passes(self, report):
        assert report["verification"]["passed"] is True

    def test_multiple_hypotheses_considered(self, report):
        assert report["metadata"]["hypotheses_considered"] >= 3


class TestOfflinePipelineRobustness:
    def test_empty_evidence_produces_no_suspect(self, scenario):
        pipeline = OfflineInvestigationPipeline(scenario["topology"])
        report = pipeline.investigate(
            case_id="empty", case_title="Empty case",
            camera_sightings=[], access_logs=[], witness_statements=[],
        )
        assert report["primary_conclusion"]["suspect_id"] is None
        assert report["confidence_assessment"]["overall_confidence"] == 0.0

    def test_topology_not_mutated_between_runs(self, scenario):
        # Regression: CameraTopology.from_dict used to pop keys from the input
        pipeline1 = OfflineInvestigationPipeline(scenario["topology"])
        pipeline2 = OfflineInvestigationPipeline(scenario["topology"])
        assert pipeline1.topology.camera_count == pipeline2.topology.camera_count
