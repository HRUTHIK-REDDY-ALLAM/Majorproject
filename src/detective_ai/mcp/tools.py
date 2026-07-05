"""MCP (Model Context Protocol) tool definitions.

Exposes evidence query capabilities as MCP tools so that agents
can access the evidence database through a standardized protocol.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from mcp.server.fastmcp import FastMCP

from detective_ai.storage.database import db

logger = logging.getLogger(__name__)

# Create the MCP server
mcp_server = FastMCP("Detective AI Evidence Tools")


@mcp_server.tool()
def query_evidence(
    evidence_type: str | None = None,
    source: str | None = None,
    limit: int = 20,
) -> str:
    """Search the evidence database.

    Args:
        evidence_type: Filter by type (video_frame, access_log, witness_statement).
        source: Filter by source (camera_id, badge_system, witness name).
        limit: Maximum results to return.
    """
    with db.session() as session:
        from detective_ai.storage.database import EvidenceRow

        q = session.query(EvidenceRow)
        if evidence_type:
            q = q.filter_by(type=evidence_type)
        if source:
            q = q.filter_by(source=source)
        rows = q.order_by(EvidenceRow.timestamp).limit(limit).all()

        results = []
        for row in rows:
            results.append({
                "id": row.id,
                "type": row.type,
                "source": row.source,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "confidence_score": row.confidence_score,
                "description": row.description,
            })

    return json.dumps(results, indent=2)


@mcp_server.tool()
def query_access_logs(
    person_id: str | None = None,
    location: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    """Query access control logs.

    Args:
        person_id: Badge holder ID to filter by.
        location: Location/door to filter by.
        start_time: ISO timestamp for range start.
        end_time: ISO timestamp for range end.
    """
    st = datetime.fromisoformat(start_time) if start_time else None
    et = datetime.fromisoformat(end_time) if end_time else None

    with db.session() as session:
        rows = db.get_access_logs(session, person_id=person_id, location=location,
                                   start_time=st, end_time=et)
        results = []
        for row in rows:
            results.append({
                "id": row.id,
                "person_id": row.person_id,
                "person_name": row.person_name,
                "location": row.location,
                "timestamp": row.timestamp.isoformat(),
                "action": row.action,
            })

    return json.dumps(results, indent=2)


@mcp_server.tool()
def get_investigation_status(case_id: str) -> str:
    """Get the current status of an investigation case.

    Args:
        case_id: The investigation case ID.
    """
    with db.session() as session:
        case = db.get_case(session, case_id)
        if not case:
            return json.dumps({"error": f"Case {case_id} not found"})

        return json.dumps({
            "id": case.id,
            "title": case.title,
            "status": case.status,
            "phase": case.phase,
            "current_round": case.current_round,
            "created_at": case.created_at.isoformat() if case.created_at else None,
        })


@mcp_server.tool()
def get_hypotheses(case_id: str) -> str:
    """Get all hypotheses for a case.

    Args:
        case_id: The investigation case ID.
    """
    with db.session() as session:
        rows = db.get_hypotheses(session, case_id)
        results = []
        for row in rows:
            results.append({
                "id": row.id,
                "title": row.title,
                "description": row.description,
                "confidence": row.confidence,
                "status": row.status,
                "rejection_reason": row.rejection_reason,
            })

    return json.dumps(results, indent=2)
