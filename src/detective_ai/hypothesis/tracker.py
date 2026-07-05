"""Multi-hypothesis tracker with branching, pruning, and audit trail.

Maintains a tree of competing hypotheses. Each hypothesis accumulates
supporting and contradicting evidence. Weak branches are pruned and
logged with rejection reasons so the final report shows its work.
"""

from __future__ import annotations

import logging
from datetime import datetime

from detective_ai.core.enums import HypothesisStatus
from detective_ai.core.models import Hypothesis, HypothesisEvidence

logger = logging.getLogger(__name__)


class HypothesisTracker:
    """Manages competing hypotheses as a branching tree."""

    def __init__(
        self,
        prune_threshold: float = 0.15,
        max_active: int = 5,
    ) -> None:
        """
        Args:
            prune_threshold: Hypotheses below this confidence are pruned.
            max_active: Maximum number of active hypotheses at any time.
        """
        self.prune_threshold = prune_threshold
        self.max_active = max_active
        self._hypotheses: dict[str, Hypothesis] = {}
        self._pruned: list[Hypothesis] = []

    # ── CRUD ──────────────────────────────────────────────────

    def add_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Register a new hypothesis."""
        self._hypotheses[hypothesis.id] = hypothesis
        logger.info(
            f"Added hypothesis '{hypothesis.title}' "
            f"(id={hypothesis.id[:8]}, confidence={hypothesis.confidence:.2f})"
        )
        self._auto_prune()
        return hypothesis.id

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        return self._hypotheses.get(hypothesis_id)

    def get_active_hypotheses(self) -> list[Hypothesis]:
        """Get all hypotheses with ACTIVE or UNDER_REVIEW status."""
        return [
            h for h in self._hypotheses.values()
            if h.status in (HypothesisStatus.ACTIVE, HypothesisStatus.UNDER_REVIEW)
        ]

    def get_all_hypotheses(self) -> list[Hypothesis]:
        """Get all hypotheses including pruned."""
        return list(self._hypotheses.values()) + self._pruned

    def get_pruned_hypotheses(self) -> list[Hypothesis]:
        """Get all pruned hypotheses with their rejection reasons."""
        return list(self._pruned)

    # ── Evidence linkage ──────────────────────────────────────

    def add_supporting_evidence(
        self,
        hypothesis_id: str,
        evidence_id: str,
        weight: float = 1.0,
        reasoning: str = "",
    ) -> None:
        """Add supporting evidence to a hypothesis."""
        h = self._hypotheses.get(hypothesis_id)
        if h is None:
            return
        h.supporting_evidence.append(
            HypothesisEvidence(
                evidence_id=evidence_id,
                relationship="supports",
                weight=weight,
                reasoning=reasoning,
            )
        )
        h.updated_at = datetime.utcnow()
        self._update_confidence(hypothesis_id)

    def add_contradicting_evidence(
        self,
        hypothesis_id: str,
        evidence_id: str,
        weight: float = 1.0,
        reasoning: str = "",
    ) -> None:
        """Add contradicting evidence to a hypothesis."""
        h = self._hypotheses.get(hypothesis_id)
        if h is None:
            return
        h.contradicting_evidence.append(
            HypothesisEvidence(
                evidence_id=evidence_id,
                relationship="contradicts",
                weight=weight,
                reasoning=reasoning,
            )
        )
        h.updated_at = datetime.utcnow()
        self._update_confidence(hypothesis_id)

    # ── Branching ─────────────────────────────────────────────

    def branch_hypothesis(
        self,
        parent_id: str,
        title: str,
        description: str,
        initial_confidence: float | None = None,
    ) -> str:
        """Create a new hypothesis branching from an existing one.

        The child inherits the parent's evidence but starts with its own
        confidence and can accumulate additional evidence independently.
        """
        parent = self._hypotheses.get(parent_id)
        if parent is None:
            raise ValueError(f"Parent hypothesis {parent_id} not found")

        child = Hypothesis(
            case_id=parent.case_id,
            parent_id=parent_id,
            title=title,
            description=description,
            confidence=initial_confidence or parent.confidence * 0.8,
            status=HypothesisStatus.ACTIVE,
            supporting_evidence=list(parent.supporting_evidence),
            contradicting_evidence=list(parent.contradicting_evidence),
        )

        self._hypotheses[child.id] = child
        logger.info(
            f"Branched hypothesis '{title}' from '{parent.title}' "
            f"(confidence={child.confidence:.2f})"
        )
        self._auto_prune()
        return child.id

    # ── Pruning ───────────────────────────────────────────────

    def prune_hypothesis(self, hypothesis_id: str, reason: str) -> None:
        """Prune a hypothesis, logging the rejection reason."""
        h = self._hypotheses.pop(hypothesis_id, None)
        if h is None:
            return
        h.status = HypothesisStatus.PRUNED
        h.rejection_reason = reason
        h.updated_at = datetime.utcnow()
        self._pruned.append(h)
        logger.info(f"Pruned hypothesis '{h.title}': {reason}")

    def _auto_prune(self) -> None:
        """Automatically prune hypotheses below threshold or excess active count."""
        active = self.get_active_hypotheses()

        # Prune by confidence threshold
        for h in active:
            if h.confidence < self.prune_threshold:
                self.prune_hypothesis(
                    h.id,
                    f"Confidence {h.confidence:.2f} below threshold {self.prune_threshold}",
                )

        # Prune excess (keep top N by confidence)
        active = self.get_active_hypotheses()
        if len(active) > self.max_active:
            active.sort(key=lambda h: h.confidence, reverse=True)
            for h in active[self.max_active :]:
                self.prune_hypothesis(
                    h.id,
                    f"Exceeded max active limit ({self.max_active}), "
                    f"lowest confidence among active ({h.confidence:.2f})",
                )

    # ── Confidence update ─────────────────────────────────────

    def _update_confidence(self, hypothesis_id: str) -> None:
        """Recalculate confidence based on evidence balance."""
        h = self._hypotheses.get(hypothesis_id)
        if h is None:
            return
        # Evidence balance weighted by source confidence
        h.confidence = round(h.evidence_balance, 3)
        self._auto_prune()

    def set_confidence(self, hypothesis_id: str, confidence: float) -> None:
        """Manually set confidence (e.g., from calibrated confidence engine)."""
        h = self._hypotheses.get(hypothesis_id)
        if h:
            h.confidence = round(max(0.0, min(1.0, confidence)), 3)
            h.updated_at = datetime.utcnow()
            self._auto_prune()

    # ── Leading hypothesis ────────────────────────────────────

    def get_leading_hypothesis(self) -> Hypothesis | None:
        """Get the hypothesis with the highest confidence."""
        active = self.get_active_hypotheses()
        if not active:
            return None
        return max(active, key=lambda h: h.confidence)

    # ── Serialization ─────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize tracker state."""
        return {
            "active": [h.model_dump() for h in self._hypotheses.values()],
            "pruned": [h.model_dump() for h in self._pruned],
            "settings": {
                "prune_threshold": self.prune_threshold,
                "max_active": self.max_active,
            },
        }

    def summary(self) -> dict:
        """Quick summary of tracker state."""
        return {
            "active_count": len(self.get_active_hypotheses()),
            "pruned_count": len(self._pruned),
            "leading": self.get_leading_hypothesis().title if self.get_leading_hypothesis() else None,
            "leading_confidence": (
                self.get_leading_hypothesis().confidence
                if self.get_leading_hypothesis()
                else 0.0
            ),
        }
