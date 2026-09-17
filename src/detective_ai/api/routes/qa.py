"""Q&A API routes: answer user questions about completed investigations."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter

from detective_ai.api.schemas import QARequest, StatusResponse
from detective_ai.storage.database import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/qa", tags=["Q&A"])

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "agents" / "prompts" / "qa.md"


def _load_qa_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


@router.post("/{case_id}", response_model=StatusResponse)
async def ask_question(case_id: str, request: QARequest):
    """Ask a natural-language question about a completed investigation.

    The system retrieves all evidence + the investigation report for the case,
    then sends the question to the LLM with the evidence as context.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    from detective_ai.config import settings
    from detective_ai.storage.database import (
        EvidenceRow, WitnessStatementRow, AccessLogRow,
    )

    # ── Validate case exists and is completed ────────────────────────────
    with db.session() as session:
        case = db.get_case(session, case_id)
        if not case:
            return StatusResponse(status="error", message=f"Case {case_id} not found")

        # Build evidence context
        all_evidence = session.query(EvidenceRow).all()
        evidence_rows = [
            r for r in all_evidence
            if (r.metadata_ or {}).get("case_id") == case_id
        ]
        if not evidence_rows:
            evidence_rows = all_evidence

        # Build comprehensive evidence context for the LLM
        context_parts = []

        # Case info
        context_parts.append(f"## Case: {case.title}")
        context_parts.append(f"Description: {case.description or 'N/A'}")
        context_parts.append(f"Status: {case.status}")

        # Investigation report (if completed)
        if case.report_data:
            report = case.report_data
            if isinstance(report, dict):
                context_parts.append("\n## Investigation Report")
                if report.get("summary"):
                    context_parts.append(f"Summary: {report['summary']}")
                if report.get("what_happened"):
                    context_parts.append(f"What happened: {report['what_happened']}")
                if report.get("key_findings"):
                    context_parts.append("Key findings:")
                    for f in report["key_findings"]:
                        context_parts.append(f"  - {f}")
                if report.get("primary_conclusion"):
                    pc = report["primary_conclusion"]
                    hyp = pc.get("hypothesis") or pc.get("title") or ""
                    context_parts.append(f"Conclusion: {hyp}")
                if report.get("things_we_are_not_sure_about"):
                    context_parts.append("Uncertainties:")
                    for u in report["things_we_are_not_sure_about"]:
                        context_parts.append(f"  - {u}")
                if report.get("timeline"):
                    context_parts.append("Timeline:")
                    for t in report["timeline"]:
                        event = t.get("event") or t.get("description") or ""
                        time = t.get("time") or t.get("timestamp") or "?"
                        context_parts.append(f"  - {time}: {event}")

        # Video analysis summaries
        video_summaries = [
            r for r in evidence_rows
            if (r.metadata_ or {}).get("is_video_summary")
        ]
        if video_summaries:
            context_parts.append("\n## Video Analysis")
            for vs in video_summaries:
                context_parts.append(str(vs.description))

        # Frame-level observations with people
        frame_obs = [
            r for r in evidence_rows
            if r.type == "video_frame"
            and not (r.metadata_ or {}).get("is_video_summary")
            and (r.metadata_ or {}).get("person_count", 0) > 0
        ]
        if frame_obs:
            context_parts.append(f"\n## Frame Observations ({len(frame_obs)} frames with people)")
            for r in frame_obs[:20]:
                context_parts.append(f"- {r.description}")

        # Witness statements
        stmts = session.query(WitnessStatementRow).all()
        if stmts:
            context_parts.append("\n## Witness Statements")
            for s in stmts[:10]:
                context_parts.append(f"- {str(s.source)}: \"{str(s.text)[:300]}\"")

        # Access logs
        logs = session.query(AccessLogRow).all()
        if logs:
            context_parts.append(f"\n## Access Control Logs")
            for log in logs[:15]:
                context_parts.append(
                    f"- {log.person_name or log.person_id} @ {log.location} "
                    f"({log.action}) at {log.timestamp}"
                )

        evidence_context = "\n".join(context_parts)

    # ── Send to LLM ──────────────────────────────────────────────────────
    try:
        system_prompt = _load_qa_prompt()
        llm = ChatGroq(
            model=settings.groq_model,
            temperature=0.2,
            api_key=settings.groq_api_key,
            max_retries=3,
            max_tokens=800,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
## Evidence and Investigation Data
{evidence_context}

## User Question
{request.question}

Answer the user's question based ONLY on the evidence and report above.
"""),
        ]

        response = llm.invoke(messages)
        response_text = str(response.content)

        # Parse JSON response
        answer_data = None
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
                answer_data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        if answer_data:
            return StatusResponse(
                status="success",
                message="Question answered",
                data={
                    "answer": answer_data.get("answer", response_text),
                    "confidence": answer_data.get("confidence", "medium"),
                    "evidence_used": answer_data.get("evidence_used", []),
                    "case_id": case_id,
                },
            )
        else:
            # LLM didn't return JSON — use raw text as answer
            return StatusResponse(
                status="success",
                message="Question answered",
                data={
                    "answer": response_text,
                    "confidence": "medium",
                    "evidence_used": [],
                    "case_id": case_id,
                },
            )

    except Exception as e:
        logger.error(f"Q&A failed for case {case_id}: {e}")
        return StatusResponse(
            status="error",
            message=f"Failed to answer question: {str(e)}",
        )
