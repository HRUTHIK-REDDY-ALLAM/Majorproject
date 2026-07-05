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

        # Update case status
        with db.session() as session:
            from datetime import datetime
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
            db.update_case(session, case_id, status="failed", phase="failed")


@router.post("/", response_model=StatusResponse)
async def start_investigation(
    request: InvestigateRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger a new investigation pipeline."""
    case_id = request.case_id or str(uuid.uuid4())

    # Create case in database
    with db.session() as session:
        db.insert_case(
            session,
            id=case_id,
            title=request.title,
            description=request.description,
            max_rounds=request.max_rounds,
        )

    # Build evidence summary from database
    evidence_summary = ""
    evidence_ids = []
    evidence_count = 0

    with db.session() as session:
        from detective_ai.storage.database import EvidenceRow
        evidence_rows = session.query(EvidenceRow).all()
        evidence_count = len(evidence_rows)
        evidence_ids = [r.id for r in evidence_rows]
        summaries = [f"- [{r.type}] {r.source} @ {r.timestamp}: {r.description}"
                     for r in evidence_rows[:50]]
        evidence_summary = "\n".join(summaries) if summaries else "No evidence ingested yet."

    # Run investigation in background
    background_tasks.add_task(
        _run_investigation_background,
        case_id=case_id,
        title=request.title,
        description=request.description,
        max_rounds=request.max_rounds,
        evidence_summary=evidence_summary,
        evidence_ids=evidence_ids,
        evidence_count=evidence_count,
    )

    return StatusResponse(
        status="success",
        message=f"Investigation started: {case_id}",
        data={"case_id": case_id, "status": "running"},
    )


@router.get("/{case_id}", response_model=StatusResponse)
async def get_investigation_status(case_id: str):
    """Get the status of an investigation."""
    with db.session() as session:
        case = db.get_case(session, case_id)
        if not case:
            return StatusResponse(status="error", message=f"Case {case_id} not found")

        return StatusResponse(
            status="success",
            message=f"Case status: {case.status}",
            data={
                "id": case.id,
                "title": case.title,
                "status": case.status,
                "phase": case.phase,
                "current_round": case.current_round,
            },
        )
