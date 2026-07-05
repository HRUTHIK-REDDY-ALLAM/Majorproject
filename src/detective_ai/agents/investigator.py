"""Investigator agent: gathers and cross-references evidence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from detective_ai.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "investigator.md"


def _load_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def investigator_node(state: AgentState, llm) -> dict[str, Any]:
    """Investigator agent node: searches and analyzes evidence.

    Queries the evidence database, cross-references findings across
    modalities, and reports agreements/contradictions.
    """
    system_prompt = _load_prompt()

    context = f"""
## Investigation Context
- Case: {state.get('case_title', 'Unknown')}
- Round: {state.get('investigation_round', 1)}

## Evidence Summary
{state.get('evidence_summary', 'No evidence available.')}

## Current Hypotheses to Evaluate
{json.dumps(state.get('hypotheses', []), indent=2, default=str)[:3000]}

## Your Task
Analyze the available evidence and report:
1. What evidence supports or contradicts each hypothesis
2. Cross-modal conflicts (e.g., video shows X but badge log shows Y)
3. Gaps that need further investigation
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context),
    ]

    response = llm.invoke(messages)
    response_text = response.content

    # Parse findings
    findings = []
    gaps_identified = []
    updated_hypotheses = list(state.get("hypotheses", []))

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
            findings = result.get("findings", [])
            gaps_identified = result.get("gaps_identified", [])

            # Update hypothesis evidence links based on findings
            for finding in findings:
                for hyp in updated_hypotheses:
                    if finding.get("supports_hypothesis") == hyp.get("id"):
                        if "supporting_evidence_notes" not in hyp:
                            hyp["supporting_evidence_notes"] = []
                        hyp["supporting_evidence_notes"].append(finding.get("summary", ""))
                    elif finding.get("contradicts_hypothesis") == hyp.get("id"):
                        if "contradicting_evidence_notes" not in hyp:
                            hyp["contradicting_evidence_notes"] = []
                        hyp["contradicting_evidence_notes"].append(finding.get("summary", ""))

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Could not parse investigator response: {e}")

    # If no hypotheses exist yet, create initial ones from the analysis
    if not updated_hypotheses:
        updated_hypotheses = [
            {
                "id": "hyp_initial_1",
                "title": "Primary Suspect Theory",
                "description": "Initial hypothesis based on evidence review",
                "confidence": 0.5,
                "status": "active",
            },
            {
                "id": "hyp_initial_2",
                "title": "Alternative Theory",
                "description": "Alternative explanation to prevent confirmation bias",
                "confidence": 0.3,
                "status": "active",
            },
        ]

    logger.info(
        f"Investigator found {len(findings)} evidence items, "
        f"{len(gaps_identified)} gaps"
    )

    return {
        "hypotheses": updated_hypotheses,
        "gaps_found": state.get("gaps_found", 0) + len(gaps_identified),
        "current_phase": "orchestrator",
        "messages": [response],
    }
