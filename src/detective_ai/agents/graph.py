"""LangGraph workflow definition for the multi-agent investigation system.

Defines the cyclic graph: Orchestrator → Specialist → Orchestrator → ...
until convergence or max rounds reached.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from detective_ai.agents.critic import critic_node
from detective_ai.agents.investigator import investigator_node
from detective_ai.agents.orchestrator import orchestrator_node
from detective_ai.agents.reporter import reporter_node
from detective_ai.agents.state import AgentState
from detective_ai.agents.trajectory import trajectory_node
from detective_ai.agents.verifier import verifier_node
from detective_ai.config import settings

logger = logging.getLogger(__name__)


def _create_llm() -> ChatGroq:
    """Create the Groq LLM instance with rate limiting."""
    return ChatGroq(
        model=settings.groq_model,
        temperature=0.1,
        api_key=settings.groq_api_key,
        max_retries=3,
    )


def _route_from_orchestrator(state: AgentState) -> str:
    """Route from orchestrator to the appropriate specialist agent."""
    phase = state.get("current_phase", "investigator")
    round_num = state.get("investigation_round", 0)
    max_rounds = state.get("max_rounds", 3)

    if not state.get("should_continue", True):
        return "end"

    if phase == "completed":
        return "end"

    if round_num > max_rounds:
        return "reporter"

    routing_map = {
        "investigator": "investigator",
        "trajectory": "trajectory",
        "critic": "critic",
        "verifier": "verifier",
        "reporter": "reporter",
    }

    return routing_map.get(phase, "investigator")


def _route_after_specialist(state: AgentState) -> str:
    """Route back to orchestrator or to end after a specialist runs."""
    phase = state.get("current_phase", "orchestrator")

    if phase == "completed":
        return "end"

    if not state.get("should_continue", True):
        return "end"

    return "orchestrator"


def build_investigation_graph() -> StateGraph:
    """Build the LangGraph investigation workflow.

    Graph topology:
        ┌──────────────────────────────────────────┐
        │                                          │
        ▼                                          │
    Orchestrator ──→ Investigator ──→ Orchestrator │
        │                                          │
        ├──→ Trajectory Agent ──→ Orchestrator ────┤
        │                                          │
        ├──→ Critic Agent ──→ Orchestrator ────────┤
        │                                          │
        ├──→ Verifier ──→ Orchestrator ────────────┤
        │                                          │
        └──→ Reporter ──→ END
    """
    llm = _create_llm()

    # Create node functions with LLM bound
    orchestrator = partial(orchestrator_node, llm=llm)
    investigator = partial(investigator_node, llm=llm)
    trajectory = partial(trajectory_node, llm=llm)
    critic = partial(critic_node, llm=llm)
    verifier = partial(verifier_node, llm=llm)
    reporter = partial(reporter_node, llm=llm)

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("investigator", investigator)
    graph.add_node("trajectory", trajectory)
    graph.add_node("critic", critic)
    graph.add_node("verifier", verifier)
    graph.add_node("reporter", reporter)

    # Set entry point
    graph.set_entry_point("orchestrator")

    # Orchestrator routes to specialist agents
    graph.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        {
            "investigator": "investigator",
            "trajectory": "trajectory",
            "critic": "critic",
            "verifier": "verifier",
            "reporter": "reporter",
            "end": END,
        },
    )

    # All specialists route back to orchestrator (except reporter → END)
    for specialist in ["investigator", "trajectory", "critic", "verifier"]:
        graph.add_conditional_edges(
            specialist,
            _route_after_specialist,
            {
                "orchestrator": "orchestrator",
                "end": END,
            },
        )

    # Reporter always ends
    graph.add_edge("reporter", END)

    return graph


def create_investigation_app():
    """Create a compiled LangGraph application ready to run."""
    graph = build_investigation_graph()
    app = graph.compile()
    logger.info("Investigation graph compiled successfully.")
    return app


def create_initial_state(
    case_id: str,
    case_title: str,
    case_description: str = "",
    evidence_summary: str = "",
    evidence_ids: list[str] | None = None,
    evidence_count: int = 0,
    max_rounds: int = 3,
) -> AgentState:
    """Create the initial state for a new investigation."""
    return AgentState(
        case_id=case_id,
        case_title=case_title,
        case_description=case_description,
        evidence_summary=evidence_summary,
        evidence_ids=evidence_ids or [],
        evidence_count=evidence_count,
        hypotheses=[],
        active_hypothesis_id=None,
        leading_hypothesis=None,
        trajectory_segments=[],
        gaps_found=0,
        gaps_filled=0,
        critic_objections=[],
        unresolved_objections=0,
        verification_results=[],
        verification_passed=False,
        report_draft=None,
        report_final=None,
        investigation_round=0,
        max_rounds=max_rounds,
        current_phase="investigator",
        should_continue=True,
        error=None,
        messages=[],
    )


async def run_investigation(
    case_id: str,
    case_title: str,
    case_description: str = "",
    evidence_summary: str = "",
    evidence_ids: list[str] | None = None,
    evidence_count: int = 0,
    max_rounds: int = 3,
) -> dict[str, Any]:
    """Run a complete investigation and return the final report.

    Args:
        case_id: Unique case identifier.
        case_title: Title of the investigation.
        case_description: Description of what to investigate.
        evidence_summary: LLM-digestible summary of all evidence.
        evidence_ids: List of evidence IDs in the database.
        evidence_count: Total number of evidence items.
        max_rounds: Maximum investigation rounds.

    Returns:
        Final investigation report dict.
    """
    app = create_investigation_app()

    initial_state = create_initial_state(
        case_id=case_id,
        case_title=case_title,
        case_description=case_description,
        evidence_summary=evidence_summary,
        evidence_ids=evidence_ids,
        evidence_count=evidence_count,
        max_rounds=max_rounds,
    )

    logger.info(f"Starting investigation: {case_title} (max {max_rounds} rounds)")

    # Run the graph
    final_state = None
    async for state in app.astream(initial_state):
        final_state = state
        # Log progress
        for node_name, node_state in state.items():
            if isinstance(node_state, dict):
                phase = node_state.get("current_phase", "")
                round_num = node_state.get("investigation_round", 0)
                logger.info(f"[{node_name}] Phase: {phase}, Round: {round_num}")

    # Extract final report
    if final_state:
        # Get the last state from any node
        for node_state in final_state.values():
            if isinstance(node_state, dict) and node_state.get("report_final"):
                return node_state["report_final"]

    return {"error": "Investigation did not produce a final report"}
