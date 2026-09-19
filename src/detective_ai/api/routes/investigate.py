"""Investigation API routes."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks

from detective_ai.agents.graph import run_investigation
from detective_ai.api.schemas import InvestigateRequest, StatusResponse
from detective_ai.storage.database import db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/investigate", tags=["Investigation"])

# In-memory investigation results (for local development)
_investigation_results: dict[str, Any] = {}


async def _run_investigation_background(
    case_id: str, title: str, description: str, max_rounds: int,
    evidence_summary: str = "", evidence_ids: list[str] | None = None,
    evidence_count: int = 0,
):
    """Background task to run the investigation pipeline."""
    try:
        # Update case status
        with db.session() as session:
            db.update_case(session, case_id, status="running", phase="investigation")

        result = await run_investigation(
            case_id=case_id,
            case_title=title,
            case_description=description,
            evidence_summary=evidence_summary,
            evidence_ids=evidence_ids,
            evidence_count=evidence_count,
            max_rounds=max_rounds,
        )

        _investigation_results[case_id] = result

        from datetime import datetime

        # run_investigation swallows pipeline errors and returns them in the
        # result, so a failure here is not an exception — check for it, or the
        # case is marked completed with nothing to show.
        if isinstance(result, dict) and result.get("error"):
            logger.error(f"Investigation {case_id} failed: {result['error']}")
            with db.session() as session:
                db.update_case(
                    session, case_id,
                    status="failed",
                    phase="failed",
                    completed_at=datetime.utcnow(),
                    report_data=result,
                )
            return

        with db.session() as session:
            db.update_case(
                session, case_id,
                status="completed",
                phase="completed",
                completed_at=datetime.utcnow(),
                report_data=result,
            )

        logger.info(f"Investigation {case_id} completed successfully")

    except Exception as e:
        logger.error(f"Investigation {case_id} failed: {e}")
        _investigation_results[case_id] = {"error": str(e)}
        with db.session() as session:
            db.update_case(
                session, case_id, status="failed", phase="failed",
                report_data={"error": str(e)},
            )


@router.post("/", response_model=StatusResponse)
async def start_investigation(
    request: InvestigateRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger a new investigation pipeline for a case."""
    case_id = request.case_id or str(uuid.uuid4())

    # Check if case already exists — read all values inside the session
    existing_status = None
    existing_title = None
    existing_description = None
    with db.session() as session:
        existing = db.get_case(session, case_id)
        if existing:
            existing_status = existing.status
            existing_title = existing.title
            existing_description = existing.description or ""

    if existing_status is not None:
        # Case exists
        if existing_status == "running":
            return StatusResponse(
                status="error",
                message=f"Investigation for case '{case_id}' is already running. Wait for it to complete.",
            )
        # Re-investigate: use provided title/description, or fall back to existing
        title = request.title or existing_title or f"Investigation {case_id[:8]}"
        description = request.description or existing_description
        # Reset case status to running
        with db.session() as session:
            db.update_case(
                session, case_id,
                title=title,
                description=description,
                status="running",
                phase="investigation",
                current_round=0,
                completed_at=None,
                report_data=None,
            )
    else:
        # New case — create it
        title = request.title or f"Investigation {case_id[:8]}"
        description = request.description or ""
        try:
            with db.session() as session:
                db.insert_case(
                    session,
                    id=case_id,
                    title=title,
                    description=description,
                    max_rounds=request.max_rounds,
                    status="running",
                    phase="investigation",
                )
        except Exception as e:
            logger.error(f"Failed to create case {case_id}: {e}")
            return StatusResponse(
                status="error",
                message=f"Failed to create case: {e}",
            )

    # Build evidence summary scoped to this case_id
    evidence_summary = ""
    evidence_ids = []
    evidence_count = 0

    with db.session() as session:
        from detective_ai.storage.database import (
            AccessLogRow,
            EvidenceRow,
            VisualDetectionRow,
            WitnessStatementRow,
        )

        # Get all evidence rows and filter by case_id in Python (SQLite JSON compat)
        all_evidence = session.query(EvidenceRow).all()
        evidence_rows = [
            r for r in all_evidence
            if (r.metadata_ or {}).get("case_id") == case_id
        ]

        # If nothing is tagged with this case, fall back only to untagged
        # evidence — never to evidence belonging to a different case.
        if not evidence_rows:
            evidence_rows = [
                r for r in all_evidence
                if not (r.metadata_ or {}).get("case_id")
            ]
            if evidence_rows:
                logger.warning(
                    f"Case {case_id}: no case-tagged evidence; "
                    f"using {len(evidence_rows)} untagged item(s)."
                )

        evidence_count = len(evidence_rows)
        evidence_ids = [r.id for r in evidence_rows]

        summaries = []

        # ── Video analysis summaries (most important — these contain the narrative)
        video_summaries = [
            r for r in evidence_rows
            if (r.metadata_ or {}).get("is_video_summary")
        ]
        for vs in video_summaries:
            summaries.append(f"### Video Analysis\n{vs.description}")

        # ── Per-window scene observations from the vision model. These are
        # the richest evidence available, so they come before raw frames.
        segment_rows = sorted(
            (r for r in evidence_rows if (r.metadata_ or {}).get("is_video_segment")),
            key=lambda r: (r.metadata_ or {}).get("segment_index", 0),
        )
        if segment_rows:
            summaries.append(
                f"\n### Observed Scene Content ({len(segment_rows)} time windows)"
            )
            for r in segment_rows:
                summaries.append(f"- {r.description}")

        # ── Frame-level detections. Only useful when no scene understanding
        # exists; otherwise they add box geometry noise to the prompt.
        if not segment_rows:
            frame_evidence = [
                r for r in evidence_rows
                if r.type == "video_frame"
                and not (r.metadata_ or {}).get("is_video_summary")
                and (r.metadata_ or {}).get("person_count", 0) > 0
            ]
            if frame_evidence:
                summaries.append(
                    f"\n### Frame-Level Detections "
                    f"({len(frame_evidence)} frames with detections, showing 15)"
                )
                for r in frame_evidence[:15]:
                    summaries.append(f"- {r.description}")

        # ── Visual detections summary
        all_detections = session.query(VisualDetectionRow).all()
        case_detections = [
            d for d in all_detections
            if d.evidence_id in evidence_ids
        ]
        if case_detections:
            person_dets = [d for d in case_detections if d.label == "person"]
            vehicle_dets = [d for d in case_detections if d.label == "vehicle"]
            cameras = sorted({str(d.camera_id) for d in case_detections})
            summaries.append(
                f"\n### Detection Statistics\n"
                f"- Total person detections: {len(person_dets)}\n"
                f"- Total vehicle detections: {len(vehicle_dets)}\n"
                f"- Cameras involved: {', '.join(cameras)}"
            )

        # ── Access logs (scoped to this case)
        access_logs = [
            r for r in session.query(AccessLogRow).all()
            if (r.metadata_ or {}).get("case_id") == case_id
        ]
        if access_logs:
            summaries.append(f"\n### Access Control Logs ({len(access_logs)} entries)")
            for log in access_logs[:10]:
                summaries.append(
                    f"- {log.person_name or log.person_id} @ {log.location} "
                    f"({log.action}) at {log.timestamp}"
                )

        # ── Witness statements (scoped to this case)
        stmts = [
            r for r in session.query(WitnessStatementRow).all()
            if (r.metadata_ or {}).get("case_id") == case_id
        ]
        if stmts:
            summaries.append(f"\n### Witness Statements ({len(stmts)} statements)")
            for s in stmts[:5]:
                summaries.append(
                    f"- {s.source} (reliability: {s.reliability_score:.0%}): "
                    f'"{s.text[:200]}"'
                )

        # ── Non-video evidence
        other_evidence = [
            r for r in evidence_rows
            if r.type != "video_frame"
        ]
        if other_evidence:
            summaries.append(f"\n### Other Evidence ({len(other_evidence)} items)")
            for r in other_evidence[:10]:
                summaries.append(f"- [{r.type}] {r.source} @ {r.timestamp}: {r.description[:150]}")

        evidence_summary = "\n".join(summaries) if summaries else "No evidence ingested yet."

    # Run investigation in background
    background_tasks.add_task(
        _run_investigation_background,
        case_id=case_id,
        title=title,
        description=description,
        max_rounds=request.max_rounds,
        evidence_summary=evidence_summary,
        evidence_ids=evidence_ids,
        evidence_count=evidence_count,
    )

    return StatusResponse(
        status="success",
        message=f"Investigation started for case: {case_id}",
        data={"case_id": case_id, "status": "running"},
    )


@router.get("/{case_id}", response_model=StatusResponse)
async def get_investigation_status(case_id: str):
    """Get the status of an investigation."""
    with db.session() as session:
        case = db.get_case(session, case_id)
        if not case:
            return StatusResponse(status="error", message=f"Case {case_id} not found")

        report = case.report_data if isinstance(case.report_data, dict) else {}
        return StatusResponse(
            status="success",
            message=f"Case status: {case.status}",
            data={
                "id": case.id,
                "title": case.title,
                "status": case.status,
                "phase": case.phase,
                "current_round": case.current_round,
                # Surfaced so the UI can say why a run failed instead of
                # sending the user to the server logs.
                "error": _friendly_error(report.get("error")),
            },
        )


def _friendly_error(raw: str | None) -> str | None:
    """Turn a raw pipeline error into something a user can act on."""
    if not raw:
        return None
    text = str(raw)
    lowered = text.lower()
    if "rate_limit" in lowered or "429" in text:
        if "per day" in lowered or "tpd" in lowered:
            return (
                "The AI provider's daily token limit has been reached. "
                "The investigation cannot run until the quota resets "
                "(usually within a few hours), or until the Groq plan is upgraded."
            )
        return (
            "The AI provider's rate limit was hit. Wait a minute and try again."
        )
    if "authentication" in lowered or "invalid api key" in lowered or "401" in text:
        return "The Groq API key was rejected. Check GROQ_API_KEY in your .env file."
    return text[:400]
