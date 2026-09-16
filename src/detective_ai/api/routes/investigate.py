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
        from detective_ai.storage.database import EvidenceRow, WitnessStatementRow, AccessLogRow

        # Get all evidence rows and filter by case_id in Python (SQLite JSON compat)
        all_evidence = session.query(EvidenceRow).all()
        evidence_rows = [
            r for r in all_evidence
            if (r.metadata_ or {}).get("case_id") == case_id
        ]

        # If no case-specific evidence, fall back to all evidence
        if not evidence_rows:
            evidence_rows = all_evidence

        evidence_count = len(evidence_rows)
        evidence_ids = [r.id for r in evidence_rows]
        summaries = [
            f"- [{r.type}] {r.source} @ {r.timestamp}: {r.description[:100]}"
            for r in evidence_rows[:20]
        ]

        # Also include witness statements for this case
        stmts = session.query(WitnessStatementRow).all()
        for s in stmts[:5]:
            summaries.append(f"- [statement] {s.source}: {s.text[:100]}")

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
