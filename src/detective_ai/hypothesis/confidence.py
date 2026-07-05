"""Calibrated confidence scoring engine.

Computes hypothesis confidence from evidence strength, validates
calibration against benchmark ground truth, and applies Platt scaling
for calibration correction.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceEngine:
    """Computes and calibrates hypothesis confidence scores."""

    def __init__(self) -> None:
        # Platt scaling parameters (learned from benchmark calibration)
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0
        self._is_calibrated: bool = False
        # History for calibration validation
        self._predictions: list[float] = []
        self._outcomes: list[int] = []  # 1 = correct, 0 = incorrect

    # ── Confidence computation ────────────────────────────────

    def compute_confidence(
        self,
        supporting_weights: list[float],
        contradicting_weights: list[float],
        source_reliabilities: list[float],
        temporal_proximities: list[float],
    ) -> float:
        """Compute a calibrated confidence score for a hypothesis.

        Args:
            supporting_weights: Weights of supporting evidence items.
            contradicting_weights: Weights of contradicting evidence items.
            source_reliabilities: Reliability scores of evidence sources (0-1).
            temporal_proximities: How close in time evidence is to the event (0-1).

        Returns:
            Confidence score (0-1).
        """
        if not supporting_weights and not contradicting_weights:
            return 0.5  # no evidence = maximum uncertainty

        # Weighted evidence balance
        sup_total = sum(
            w * r * t
            for w, r, t in zip(
                supporting_weights,
                source_reliabilities[: len(supporting_weights)],
                temporal_proximities[: len(supporting_weights)],
            )
        ) if supporting_weights else 0.0

        con_total = sum(
            w * r * t
            for w, r, t in zip(
                contradicting_weights,
                source_reliabilities[len(supporting_weights):],
                temporal_proximities[len(supporting_weights):],
            )
        ) if contradicting_weights else 0.0

        # Bayesian-inspired combination
        total = sup_total + con_total
        if total == 0:
            raw_confidence = 0.5
        else:
            raw_confidence = sup_total / total

        # Volume bonus: more evidence = more certain
        evidence_count = len(supporting_weights) + len(contradicting_weights)
        volume_factor = 1.0 - (1.0 / (1.0 + evidence_count * 0.3))
        raw_confidence = 0.5 + (raw_confidence - 0.5) * volume_factor

        # Apply Platt scaling if calibrated
        if self._is_calibrated:
            raw_confidence = self._platt_scale(raw_confidence)

        return round(max(0.0, min(1.0, raw_confidence)), 3)

    # ── Platt scaling calibration ─────────────────────────────

    def _platt_scale(self, raw: float) -> float:
        """Apply Platt scaling: sigmoid(a * raw + b)."""
        z = self._platt_a * raw + self._platt_b
        return 1.0 / (1.0 + math.exp(-z))

    def calibrate(
        self,
        predicted_confidences: list[float],
        actual_outcomes: list[int],
    ) -> dict[str, float]:
        """Calibrate the confidence engine using benchmark results.

        Fits Platt scaling parameters to minimize calibration error.

        Args:
            predicted_confidences: Raw confidence scores.
            actual_outcomes: Binary outcomes (1 = hypothesis was correct).

        Returns:
            Dict with calibration metrics.
        """
        if len(predicted_confidences) < 5:
            logger.warning("Not enough data for calibration (need >= 5 samples)")
            return {"error": "insufficient_data"}

        preds = np.array(predicted_confidences)
        actuals = np.array(actual_outcomes)

        # Fit Platt scaling via log-likelihood
        def neg_log_likelihood(params):
            a, b = params
            z = a * preds + b
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.sum(actuals * np.log(p) + (1 - actuals) * np.log(1 - p))

        from scipy.optimize import minimize

        result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method="Nelder-Mead")
        self._platt_a, self._platt_b = result.x
        self._is_calibrated = True

        # Compute metrics
        ece = self.expected_calibration_error(predicted_confidences, actual_outcomes)

        logger.info(
            f"Calibration complete: a={self._platt_a:.3f}, b={self._platt_b:.3f}, ECE={ece:.4f}"
        )

        return {
            "platt_a": self._platt_a,
            "platt_b": self._platt_b,
            "ece_before": ece,
            "sample_count": len(predicted_confidences),
        }

    # ── Calibration metrics ───────────────────────────────────

    @staticmethod
    def expected_calibration_error(
        confidences: list[float],
        outcomes: list[int],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Measures whether 80%-confidence claims are correct ~80% of the time.

        Args:
            confidences: Predicted confidence scores.
            outcomes: Binary outcomes.
            n_bins: Number of calibration bins.

        Returns:
            ECE value (lower is better).
        """
        confs = np.array(confidences)
        outs = np.array(outcomes)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(confs)

        for i in range(n_bins):
            mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
            bin_size = mask.sum()
            if bin_size == 0:
                continue
            bin_accuracy = outs[mask].mean()
            bin_confidence = confs[mask].mean()
            ece += (bin_size / total) * abs(bin_accuracy - bin_confidence)

        return float(ece)

    @staticmethod
    def reliability_diagram(
        confidences: list[float],
        outcomes: list[int],
        n_bins: int = 10,
    ) -> dict[str, list[float]]:
        """Compute data for a reliability diagram (calibration curve).

        Returns:
            Dict with 'bin_centers', 'accuracies', 'counts'.
        """
        confs = np.array(confidences)
        outs = np.array(outcomes)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)

        centers = []
        accuracies = []
        counts = []

        for i in range(n_bins):
            mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
            bin_size = mask.sum()
            center = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
            centers.append(float(center))
            counts.append(int(bin_size))
            if bin_size > 0:
                accuracies.append(float(outs[mask].mean()))
            else:
                accuracies.append(0.0)

        return {
            "bin_centers": centers,
            "accuracies": accuracies,
            "counts": counts,
        }

    # ── Record predictions for later calibration ──────────────

    def record_prediction(self, confidence: float, outcome: int) -> None:
        """Record a prediction–outcome pair for calibration tracking."""
        self._predictions.append(confidence)
        self._outcomes.append(outcome)

    def auto_calibrate(self) -> dict | None:
        """Calibrate from accumulated predictions if enough data."""
        if len(self._predictions) >= 10:
            return self.calibrate(self._predictions, self._outcomes)
        return None
