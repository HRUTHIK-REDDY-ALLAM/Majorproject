"""Critic agent: adversarial review of the leading hypothesis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from detective_ai.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "critic.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def critic_node(state: AgentState, llm) -> dict[str, Any]:
    """Critic agent node: adversarially reviews the leading hypothesis.

    Systematically attacks the leading theory to identify weaknesses,
    alternative explanations, and unsupported conclusions.
    """
    system_prompt = _load_prompt()

    leading = state.get("leading_hypothesis", {})
    if not leading:
        # Find the highest-confidence active hypothesis
        hypotheses = state.get("hypotheses", [])
        active = [h for h in hypotheses if h.get("status") == "active"]
        if active:
            leading = max(active, key=lambda h: h.get("confidence", 0))

    context = f"""
## Hypothesis Under Review
{json.dumps(leading, indent=2, default=str)}

## All Evidence Summary
{state.get('evidence_summary', 'No evidence summary available.')}

## Trajectory Segments (including inferred)
{json.dumps(state.get('trajectory_segments', [])[:10], indent=2, default=str)}

## Other Active Hypotheses
{json.dumps([h for h in state.get('hypotheses', []) if h.get('id') != leading.get('id') and h.get('status') == 'active'], indent=2, default=str)}

YOUR TASK: Attack the leading hypothesis. Find weaknesses, alternative explanations,
and unsupported conclusions. Be thorough and aggressive.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    response_text = response.content

    # Parse objections
    objections = list(state.get("critic_objections", []))
    confidence_adjustment = 0.0

    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "{" in response_text:
            start = response_text.index("{")
            end = response_text.rindex("}") + 1
            json_str = response_text[start:end]
        else:
            json_str = ""

        if json_str:
            result = json.loads(json_str)
            new_objections = result.get("objections", [])
            confidence_adjustment = result.get("recommended_confidence_adjustment", 0)

            for obj in new_objections:
                objections.append({
                    "hypothesis_id": leading.get("id", ""),
                    "severity": obj.get("severity", "MINOR"),
                    "target_claim": obj.get("target_claim", ""),
                    "objection_text": obj.get("objection", ""),
                    "alternative_explanation": obj.get("alternative_explanation", ""),
                    "resolved": False,
                })

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse critic response: {e}")

    # Update hypothesis confidence based on critic review
    updated_hypotheses = list(state.get("hypotheses", []))
    for hyp in updated_hypotheses:
        if hyp.get("id") == leading.get("id"):
            current_conf = hyp.get("confidence", 0.5)
            new_conf = max(0.05, min(0.99, current_conf + confidence_adjustment))
            hyp["confidence"] = round(new_conf, 3)
            break

    unresolved = sum(1 for o in objections if not o.get("resolved", False))

    logger.info(
        f"Critic review: {len(objections)} total objections, "
        f"{unresolved} unresolved, confidence adjustment={confidence_adjustment:+.2f}"
    )

    return {
        "critic_objections": objections,
        "unresolved_objections": unresolved,
        "hypotheses": updated_hypotheses,
        "current_phase": "orchestrator",
        "messages": [response],
    }
