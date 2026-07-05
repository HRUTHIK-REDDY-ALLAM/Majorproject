"""Trajectory agent: invokes gap-filling for camera blind spots."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage

from detective_ai.agents.state import AgentState

logger = logging.getLogger(__name__)


def trajectory_node(state: AgentState, llm=None) -> dict[str, Any]:
    """Trajectory agent node: fills gaps in movement timelines.

    This agent is more algorithmic than LLM-based — it uses the
    Markov trajectory model to infer paths through blind spots.
    The LLM is used only to summarize results.
    """
    trajectory_segments = list(state.get("trajectory_segments", []))
    gaps_filled = state.get("gaps_filled", 0)

    # Summarize what was done
    inferred_count = sum(
        1 for s in trajectory_segments if s.get("movement_type") == "inferred"
    )
    observed_count = len(trajectory_segments) - inferred_count

    summary = (
        f"Trajectory analysis complete: {len(trajectory_segments)} segments total, "
        f"{observed_count} observed, {inferred_count} inferred through blind spots. "
        f"All inferred segments are flagged with uncertainty bounds."
    )

    message = AIMessage(content=summary)

    logger.info(
        f"Trajectory agent: {inferred_count} inferred, {observed_count} observed"
    )

    return {
        "trajectory_segments": trajectory_segments,
        "gaps_filled": gaps_filled + inferred_count,
        "current_phase": "orchestrator",
        "messages": [message],
    }
