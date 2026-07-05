"""Report and counterfactual API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from detective_ai.api.schemas import CounterfactualRequest, StatusResponse
from detective_ai.storage.database import db

logger = logging.getLogger(__name__)

report_router = APIRouter(prefix="/api/v1/report", tags=["Reports"])
counterfactual_router = APIRouter(prefix="/api/v1/counterfactual", tags=["Counterfactual"])


@report_router.get("/{case_id}")
async def get_report(case_id: str):
    """Retrieve the final investigation report for a case."""
    with db.session() as session:
        case = db.get_case(session, case_id)
        if not case:
            return StatusResponse(status="error", message=f"Case {case_id} not found")

        if case.status != "completed":
            return StatusResponse(
                status="pending",
                message=f"Investigation not yet completed (status: {case.status})",
                data={"status": case.status, "phase": case.phase},
            )

        return StatusResponse(
            status="success",
            message="Report retrieved",
            data={"report": case.report_data},
        )


@counterfactual_router.post("/")
async def run_counterfactual(request: CounterfactualRequest):
    """Run a counterfactual analysis: 'What if evidence X were false?'"""
    from detective_ai.core.enums import HypothesisStatus
    from detective_ai.core.models import Hypothesis
    from detective_ai.hypothesis.counterfactual import CounterfactualEngine
    from detective_ai.hypothesis.tracker import HypothesisTracker

    with db.session() as session:
        case = db.get_case(session, case_id=request.case_id)
        if not case:
            return StatusResponse(status="error", message="Case not found")

        # Rebuild tracker from database
        tracker = HypothesisTracker()
        hypothesis_rows = db.get_hypotheses(session, request.case_id)

        for row in hypothesis_rows:
            h = Hypothesis(
                id=row.id,
                case_id=row.case_id,
                parent_id=row.parent_id,
                title=row.title,
                description=row.description,
                confidence=row.confidence,
                status=HypothesisStatus(row.status),
            )
            tracker.add_hypothesis(h)

    engine = CounterfactualEngine(tracker)
    result = engine.explore(request.removed_evidence_id)

    return StatusResponse(
        status="success",
        message="Counterfactual analysis complete",
        data=result,
    )
