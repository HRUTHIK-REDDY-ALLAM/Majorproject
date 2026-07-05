"""LangGraph shared state definition for the multi-agent system."""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state passed between all agents in the investigation graph.

    Every node in the LangGraph workflow reads from and writes to this state.
    """

    # ── Case metadata ─────────────────────────────────────────
    case_id: str
    case_title: str
    case_description: str

    # ── Evidence ──────────────────────────────────────────────
    evidence_summary: str  # LLM-digestible summary of all evidence
    evidence_ids: list[str]
    evidence_count: int

    # ── Hypotheses ────────────────────────────────────────────
    hypotheses: list[dict[str, Any]]  # serialized Hypothesis objects
    active_hypothesis_id: str | None
    leading_hypothesis: dict[str, Any] | None

    # ── Trajectories ──────────────────────────────────────────
    trajectory_segments: list[dict[str, Any]]
    gaps_found: int
    gaps_filled: int

    # ── Critic ────────────────────────────────────────────────
    critic_objections: list[dict[str, Any]]
    unresolved_objections: int

    # ── Verification ──────────────────────────────────────────
    verification_results: list[dict[str, Any]]
    verification_passed: bool

    # ── Report ────────────────────────────────────────────────
    report_draft: dict[str, Any] | None
    report_final: dict[str, Any] | None

    # ── Control flow ──────────────────────────────────────────
    investigation_round: int
    max_rounds: int
    current_phase: str
    should_continue: bool
    error: str | None

    # ── Message history ───────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
