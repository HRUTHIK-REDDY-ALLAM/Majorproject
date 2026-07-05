"""Verifier agent: audits the draft report for correctness."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from detective_ai.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "verifier.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def verifier_node(state: AgentState, llm) -> dict[str, Any]:
    """Verifier agent node: checks that every claim is evidence-backed.

    Performs a formal correctness check distinct from the critic's
    adversarial role. Focuses on citation integrity and logical consistency.
    """
    system_prompt = _load_prompt()

    context = f"""
## Report Draft to Verify
{json.dumps(state.get('report_draft'), indent=2, default=str)[:4000]}

## Available Evidence IDs
{json.dumps(state.get('evidence_ids', [])[:50], default=str)}

## Hypotheses
{json.dumps(state.get('hypotheses', []), indent=2, default=str)[:2000]}

## Trajectory Segments
{json.dumps(state.get('trajectory_segments', [])[:10], indent=2, default=str)}

YOUR TASK: Verify every claim in the report draft traces to cited evidence.
Flag any unsupported claims or inferential leaps.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    response_text = response.content

    verification_results = []
    verification_passed = True

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
            verification_results = result.get("results", [])
            verification_passed = result.get("overall_status", "FAILED") == "PASSED"

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse verifier response: {e}")

    logger.info(
        f"Verification {'PASSED' if verification_passed else 'FAILED'}: "
        f"{len(verification_results)} claims checked"
    )

    return {
        "verification_results": verification_results,
        "verification_passed": verification_passed,
        "current_phase": "reporter" if verification_passed else "orchestrator",
        "messages": [response],
    }
