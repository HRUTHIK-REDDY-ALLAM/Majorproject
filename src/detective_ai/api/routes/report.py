"""Report API routes."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from detective_ai.api.schemas import StatusResponse
from detective_ai.storage.database import db

logger = logging.getLogger(__name__)

report_router = APIRouter(prefix="/api/v1/report", tags=["Reports"])


@report_router.get("/{case_id}")
async def get_report(case_id: str):
    """Retrieve the final investigation report for a case."""
    with db.session() as session:
        case = db.get_case(session, case_id)
        if not case:
            return StatusResponse(status="error", message=f"Case {case_id} not found")

        if case.status != "completed":
            from detective_ai.api.routes.investigate import _friendly_error

            report = case.report_data if isinstance(case.report_data, dict) else {}
            reason = _friendly_error(report.get("error"))
            message = (
                f"Investigation failed: {reason}" if reason
                else f"Investigation not yet completed (status: {case.status})"
            )
            return StatusResponse(
                status="pending",
                message=message,
                data={"status": case.status, "phase": case.phase, "error": reason},
            )

        return StatusResponse(
            status="success",
            message="Report retrieved",
            data={"report": case.report_data},
        )
