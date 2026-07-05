"""Orchestrator agent: lead investigator that coordinates the investigation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from detective_ai.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "orchestrator.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def orchestrator_node(state: AgentState, llm) -> dict[str, Any]:
    """Orchestrator agent node: analyzes state and decides next action.

    This is the central routing node in the investigation graph.
    It examines current hypotheses, evidence, and investigation round
    to determine which specialist agent should run next.
    """
    system_prompt = _load_prompt()
    round_num = state.get("investigation_round", 0)
    max_rounds = state.get("max_rounds", 3)

    # Build context for the LLM
    context = f"""
## Current Investigation State
- Case: {state.get('case_title', 'Unknown')}
- Round: {round_num + 1} of {max_rounds}
- Evidence items: {state.get('evidence_count', 0)}
- Active hypotheses: {len([h for h in state.get('hypotheses', []) if h.get('status') == 'active'])}
- Gaps found: {state.get('gaps_found', 0)}, Gaps filled: {state.get('gaps_filled', 0)}
- Unresolved critic objections: {state.get('unresolved_objections', 0)}
- Current phase: {state.get('current_phase', 'unknown')}

## Evidence Summary
{state.get('evidence_summary', 'No evidence summary available.')}

## Current Hypotheses
{json.dumps(state.get('hypotheses', []), indent=2, default=str)[:3000]}

## Critic Objections (if any)
{json.dumps(state.get('critic_objections', []), indent=2, default=str)[:2000]}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    response_text = response.content

    # Parse the LLM's routing decision
    next_agent = "investigator"  # default

    try:
        # Try to extract JSON from response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "{" in response_text:
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            json_str = response_text[start:end]
        else:
            json_str = ""

        if json_str:
            decision = json.loads(json_str)
            next_agent = decision.get("next_agent", "investigator")

            # Process hypothesis updates
            for update in decision.get("hypothesis_updates", []):
                logger.info(f"Hypothesis update: {update}")

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse orchestrator JSON response: {e}")

    # Determine next phase based on round and state
    if round_num >= max_rounds - 1:
        # Final round — go to verification then reporting
        if not state.get("verification_results"):
            next_agent = "verifier"
        else:
            next_agent = "reporter"
    elif round_num == 0 and not state.get("hypotheses"):
        next_agent = "investigator"
    elif state.get("gaps_found", 0) > state.get("gaps_filled", 0):
        next_agent = "trajectory"

    leading = state.get("leading_hypothesis")
    if leading and leading.get("confidence", 0) > 0.7 and not state.get("critic_objections"):
        next_agent = "critic"

    logger.info(f"Orchestrator routing to: {next_agent} (round {round_num + 1})")

    return {
        "current_phase": next_agent,
        "investigation_round": round_num + 1,
        "messages": [response],
    }
