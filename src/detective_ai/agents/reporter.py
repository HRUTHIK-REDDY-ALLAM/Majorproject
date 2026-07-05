"""Reporter agent: assembles the final investigation report."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from detective_ai.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "reporter.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def reporter_node(state: AgentState, llm) -> dict[str, Any]:
    """Reporter agent node: assembles the final investigation report.

    Produces a structured report with cited claims, confidence scores,
    rejected alternatives, and unresolved objections.
    """
    system_prompt = _load_prompt()

    context = f"""
## Case Details
- Title: {state.get('case_title', 'Unknown Case')}
- Description: {state.get('case_description', '')}

## Evidence Summary
{state.get('evidence_summary', 'No evidence available.')}

## Final Hypotheses
{json.dumps(state.get('hypotheses', []), indent=2, default=str)[:3000]}

## Trajectory Segments
{json.dumps(state.get('trajectory_segments', []), indent=2, default=str)[:2000]}

## Critic Objections
{json.dumps(state.get('critic_objections', []), indent=2, default=str)[:2000]}

## Verification Results
{json.dumps(state.get('verification_results', []), indent=2, default=str)[:1500]}

## Investigation Rounds Completed: {state.get('investigation_round', 0)}

YOUR TASK: Assemble the final investigation report following the required structure.
Every claim MUST cite evidence. Include rejected alternatives and unresolved objections.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    response_text = response.content

    # Parse report
    report = None
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
            report = json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse reporter JSON: {e}")
        report = {
            "title": f"Investigation Report: {state.get('case_title', 'Unknown')}",
            "summary": response_text[:2000],
            "raw_response": response_text,
        }

    # Enrich report with metadata
    if report:
        report["metadata"] = {
            "investigation_rounds": state.get("investigation_round", 0),
            "evidence_items_analyzed": state.get("evidence_count", 0),
            "hypotheses_considered": len(state.get("hypotheses", [])),
            "hypotheses_rejected": sum(
                1 for h in state.get("hypotheses", []) if h.get("status") == "pruned"
            ),
            "critic_objections_total": len(state.get("critic_objections", [])),
            "unresolved_objections": state.get("unresolved_objections", 0),
            "verification_passed": state.get("verification_passed", False),
        }

    logger.info("Final report generated.")

    return {
        "report_final": report,
        "current_phase": "completed",
        "should_continue": False,
        "messages": [response],
    }
