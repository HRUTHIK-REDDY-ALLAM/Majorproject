"""Benchmark evaluator: computes accuracy and calibration metrics.

Evaluates the Detective AI system against ground-truth synthetic scenarios.
Reports: timeline accuracy, suspect identification precision, confidence
calibration (ECE), critic effectiveness, and time-to-resolution.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from detective_ai.hypothesis.confidence import ConfidenceEngine

logger = logging.getLogger(__name__)


class BenchmarkEvaluator:
    """Evaluates investigation results against ground truth."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.confidence_engine = ConfidenceEngine()

    def evaluate_scenario(
        self,
        scenario: dict[str, Any],
        investigation_result: dict[str, Any],
        resolution_time_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Evaluate a single scenario's investigation result against ground truth.

        Args:
            scenario: The ground truth scenario from the generator.
            investigation_result: The system's investigation report.
            resolution_time_seconds: Wall-clock time taken.

        Returns:
            Dict of evaluation metrics for this scenario.
        """
        ground_truth = scenario.get("ground_truth", {})
        true_suspect = ground_truth.get("suspect", {})

        # 1. Suspect identification accuracy
        predicted_suspect = self._extract_predicted_suspect(investigation_result)
        suspect_correct = (
            predicted_suspect.get("id") == true_suspect.get("id")
            if predicted_suspect
            else False
        )

        # 2. Timeline reconstruction accuracy (IoU-based)
        true_path = ground_truth.get("suspect_path", [])
        predicted_timeline = investigation_result.get("timeline", [])
        timeline_accuracy = self._compute_timeline_accuracy(
            true_path, predicted_timeline
        )

        # 3. Confidence calibration
        reported_confidence = investigation_result.get(
            "confidence_assessment", {}
        ).get("overall_confidence", 0.5)

        # 4. Critic effectiveness (did it catch the false leads?)
        false_leads = scenario.get("false_leads", [])
        objections = investigation_result.get("unresolved_objections", [])
        critic_effectiveness = self._compute_critic_effectiveness(
            false_leads, objections, investigation_result
        )

        # 5. False positive rate
        false_positive = not suspect_correct

        result = {
            "scenario_id": scenario.get("id"),
            "scenario_type": scenario.get("scenario_type"),
            "suspect_correct": suspect_correct,
            "suspect_predicted": predicted_suspect.get("id") if predicted_suspect else None,
            "suspect_actual": true_suspect.get("id"),
            "timeline_accuracy": round(timeline_accuracy, 3),
            "reported_confidence": round(reported_confidence, 3),
            "critic_effectiveness": round(critic_effectiveness, 3),
            "false_positive": false_positive,
            "resolution_time_seconds": round(resolution_time_seconds, 1),
            "hypotheses_considered": investigation_result.get("metadata", {}).get(
                "hypotheses_considered", 0
            ),
            "hypotheses_rejected": investigation_result.get("metadata", {}).get(
                "hypotheses_rejected", 0
            ),
        }

        # Record for calibration
        self.confidence_engine.record_prediction(
            reported_confidence, 1 if suspect_correct else 0
        )

        self.results.append(result)
        return result

    def compute_aggregate_metrics(self) -> dict[str, Any]:
        """Compute aggregate metrics across all evaluated scenarios."""
        if not self.results:
            return {"error": "No results to evaluate"}

        n = len(self.results)
        correct = sum(1 for r in self.results if r["suspect_correct"])
        confidences = [r["reported_confidence"] for r in self.results]
        outcomes = [1 if r["suspect_correct"] else 0 for r in self.results]

        # Confidence calibration
        ece = ConfidenceEngine.expected_calibration_error(confidences, outcomes)
        reliability = ConfidenceEngine.reliability_diagram(confidences, outcomes)

        metrics = {
            "total_scenarios": n,
            "suspect_identification": {
                "accuracy": round(correct / n, 3),
                "correct": correct,
                "false_positives": sum(1 for r in self.results if r["false_positive"]),
                "precision": round(correct / max(1, n), 3),
            },
            "timeline_accuracy": {
                "mean": round(np.mean([r["timeline_accuracy"] for r in self.results]), 3),
                "std": round(np.std([r["timeline_accuracy"] for r in self.results]), 3),
            },
            "confidence_calibration": {
                "expected_calibration_error": round(ece, 4),
                "mean_confidence": round(np.mean(confidences), 3),
                "mean_accuracy": round(np.mean(outcomes), 3),
                "reliability_diagram": reliability,
            },
            "critic_effectiveness": {
                "mean": round(
                    np.mean([r["critic_effectiveness"] for r in self.results]), 3
                ),
            },
            "resolution_time": {
                "mean_seconds": round(
                    np.mean([r["resolution_time_seconds"] for r in self.results]), 1
                ),
                "median_seconds": round(
                    np.median([r["resolution_time_seconds"] for r in self.results]), 1
                ),
            },
        }

        return metrics

    def _extract_predicted_suspect(
        self, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract the predicted suspect from the investigation result."""
        conclusion = result.get("primary_conclusion", {})
        if conclusion:
            # Try to extract suspect info from conclusion text
            hypothesis = conclusion.get("hypothesis", "")
            # Simple heuristic: look for person IDs
            for person_id in ["EMP001", "EMP002", "EMP003", "EMP004", "EMP005", "VIS001", "VIS002"]:
                if person_id in hypothesis or person_id in str(conclusion):
                    return {"id": person_id}
        return {}

    def _compute_timeline_accuracy(
        self,
        true_path: list[dict],
        predicted_timeline: list[dict],
    ) -> float:
        """Compute timeline reconstruction accuracy using IoU-like metric."""
        if not true_path or not predicted_timeline:
            return 0.0

        true_cameras = set(p.get("camera_id", "") for p in true_path)
        predicted_cameras = set()
        for event in predicted_timeline:
            # Extract camera references from timeline events
            event_text = str(event)
            for cam in true_cameras:
                if cam in event_text:
                    predicted_cameras.add(cam)

        if not true_cameras:
            return 0.0

        intersection = true_cameras & predicted_cameras
        union = true_cameras | predicted_cameras

        return len(intersection) / len(union) if union else 0.0

    def _compute_critic_effectiveness(
        self,
        false_leads: list[dict],
        objections: list[dict],
        result: dict[str, Any],
    ) -> float:
        """Compute what proportion of planted false leads the critic caught."""
        if not false_leads:
            return 1.0  # no false leads = critic has nothing to find

        caught = 0
        all_text = json.dumps(objections) + json.dumps(
            result.get("alternative_hypotheses", [])
        )

        for lead in false_leads:
            # Check if the false lead was mentioned in objections or rejected hypotheses
            implicated = lead.get("person_implicated", {})
            if implicated.get("name", "NONE") in all_text:
                caught += 1

        return caught / len(false_leads)

    def save_results(self, output_path: str | Path) -> None:
        """Save evaluation results to JSON."""
        output = {
            "individual_results": self.results,
            "aggregate_metrics": self.compute_aggregate_metrics(),
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Results saved to {path}")
