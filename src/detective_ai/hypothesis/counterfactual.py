"""Counterfactual exploration engine.

"What if X were false?" — removes a specified evidence item and re-runs
hypothesis tracking to show which conclusions change and by how much.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from detective_ai.hypothesis.tracker import HypothesisTracker

logger = logging.getLogger(__name__)


class CounterfactualEngine:
    """Enables sensitivity analysis by removing evidence and re-evaluating."""

    def __init__(self, tracker: HypothesisTracker) -> None:
        self._original_tracker = tracker

    def explore(self, removed_evidence_id: str) -> dict[str, Any]:
        """Re-run hypothesis tracking with a specific evidence item removed.

        Args:
            removed_evidence_id: The evidence ID to treat as "false" / absent.

        Returns:
            Dict showing:
            - original leading hypothesis and confidence
            - counterfactual leading hypothesis and confidence
            - per-hypothesis confidence changes
            - whether the conclusion changed
        """
        # Snapshot original state
        original_leading = self._original_tracker.get_leading_hypothesis()
        original_state = {}
        for h in self._original_tracker.get_all_hypotheses():
            original_state[h.id] = {
                "title": h.title,
                "confidence": h.confidence,
                "status": h.status.value,
            }

        # Deep-copy the tracker
        cf_tracker = copy.deepcopy(self._original_tracker)

        # Remove the evidence from all hypotheses
        for h_id in list(cf_tracker._hypotheses.keys()):
            h = cf_tracker._hypotheses[h_id]
            h.supporting_evidence = [
                e for e in h.supporting_evidence if e.evidence_id != removed_evidence_id
            ]
            h.contradicting_evidence = [
                e for e in h.contradicting_evidence if e.evidence_id != removed_evidence_id
            ]
            # Recalculate confidence
            cf_tracker._update_confidence(h_id)

        # Get counterfactual results
        cf_leading = cf_tracker.get_leading_hypothesis()
        cf_state = {}
        for h in cf_tracker.get_all_hypotheses():
            cf_state[h.id] = {
                "title": h.title,
                "confidence": h.confidence,
                "status": h.status.value,
            }

        # Compute diffs
        changes = []
        for h_id in set(list(original_state.keys()) + list(cf_state.keys())):
            orig = original_state.get(h_id, {})
            cf = cf_state.get(h_id, {})
            if orig.get("confidence") != cf.get("confidence"):
                changes.append({
                    "hypothesis_id": h_id,
                    "title": orig.get("title") or cf.get("title"),
                    "original_confidence": orig.get("confidence", 0),
                    "counterfactual_confidence": cf.get("confidence", 0),
                    "delta": round(
                        cf.get("confidence", 0) - orig.get("confidence", 0), 3
                    ),
                    "status_changed": orig.get("status") != cf.get("status"),
                })

        conclusion_changed = (
            (original_leading is None) != (cf_leading is None)
            or (
                original_leading is not None
                and cf_leading is not None
                and original_leading.id != cf_leading.id
            )
        )

        result = {
            "removed_evidence_id": removed_evidence_id,
            "conclusion_changed": conclusion_changed,
            "original_leading": {
                "id": original_leading.id if original_leading else None,
                "title": original_leading.title if original_leading else None,
                "confidence": original_leading.confidence if original_leading else 0,
            },
            "counterfactual_leading": {
                "id": cf_leading.id if cf_leading else None,
                "title": cf_leading.title if cf_leading else None,
                "confidence": cf_leading.confidence if cf_leading else 0,
            },
            "changes": changes,
        }

        logger.info(
            f"Counterfactual analysis: removed evidence {removed_evidence_id[:8]}..., "
            f"conclusion_changed={conclusion_changed}"
        )
        return result
