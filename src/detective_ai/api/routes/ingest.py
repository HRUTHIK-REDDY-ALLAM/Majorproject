"""Evidence ingestion API routes."""

from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, UploadFile

from detective_ai.api.schemas import IngestLogsRequest, IngestStatementRequest, StatusResponse
from detective_ai.ingestion.log_processor import process_access_logs
from detective_ai.ingestion.statement_processor import process_statement

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


@router.post("/video", response_model=StatusResponse)
async def ingest_video(
    file: UploadFile = File(...),
    camera_id: str = Form(""),
    start_time: str = Form(""),
    case_id: str = Form(""),
):
    """Upload and process a video file for evidence extraction."""
    import uuid
    from datetime import datetime

    from detective_ai.ingestion.video_processor import process_video
    from detective_ai.storage.database import db

    # Auto-generate camera_id if not provided
    if not camera_id:
        camera_id = f"cam_{uuid.uuid4().hex[:8]}"

    # Auto-generate case_id if not provided
    if not case_id:
        case_id = str(uuid.uuid4())

    # Default start_time to now if not provided
    if not start_time:
        parsed_start_time = datetime.utcnow()
    else:
        parsed_start_time = datetime.fromisoformat(start_time)

    # Ensure a case row exists for this case_id (create if missing)
    filename_stem = Path(file.filename or "video").stem
    with db.session() as session:
        existing = db.get_case(session, case_id)
        if not existing:
            db.insert_case(
                session,
                id=case_id,
                title=f"Case – {filename_stem}",
                description=f"Auto-created when video '{file.filename}' was ingested.",
                status="evidence_ingested",
                phase="ingestion",
            )

    # Save uploaded file temporarily
    suffix = Path(file.filename or "video.mp4").suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        evidence_items = process_video(
            video_path=tmp_path,
            camera_id=camera_id,
            start_time=parsed_start_time,
            case_id=case_id,
        )
        # Update case evidence count
        with db.session() as session:
            db.update_case(
                session, case_id,
                description=(
                    f"Video '{file.filename}' ingested — "
                    f"{len(evidence_items)} frames extracted."
                ),
            )
        return StatusResponse(
            status="success",
            message=f"Video uploaded — {len(evidence_items)} frames stored",
            data={
                "evidence_count": len(evidence_items),
                "camera_id": camera_id,
                "case_id": case_id,
            },
        )
    except Exception as e:
        logger.error(f"Video ingestion failed: {e}")
        return StatusResponse(status="error", message=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/logs", response_model=StatusResponse)
async def ingest_logs(request: IngestLogsRequest):
    """Upload and process access control logs."""
    try:
        entries = process_access_logs(
            content=request.content,
            format=request.format,
            source=request.source,
            case_id=request.case_id,
        )
        return StatusResponse(
            status="success",
            message=f"Processed {len(entries)} access log entries",
            data={"entry_count": len(entries)},
        )
    except Exception as e:
        logger.error(f"Log ingestion failed: {e}")
        return StatusResponse(status="error", message=str(e))


@router.post("/statements", response_model=StatusResponse)
async def ingest_statement(request: IngestStatementRequest):
    """Upload and process a witness statement."""
    import uuid

    from detective_ai.storage.database import db

    try:
        # Auto-generate case_id if not provided
        case_id = request.case_id or str(uuid.uuid4())

        # Ensure a case row exists for this case_id
        with db.session() as session:
            existing = db.get_case(session, case_id)
            if not existing:
                db.insert_case(
                    session,
                    id=case_id,
                    title=f"Case – statement from {request.source}",
                    description=f"Auto-created when witness statement from '{request.source}' was ingested.",
                    status="evidence_ingested",
                    phase="ingestion",
                )

        statement = process_statement(
            text=request.text,
            source=request.source,
            statement_time=request.timestamp,
            event_time=request.event_time,
            reliability_score=request.reliability_score,
            case_id=case_id,
        )
        return StatusResponse(
            status="success",
            message=f"Statement from {request.source} stored",
            data={"statement_id": statement.id, "case_id": case_id},
        )
    except Exception as e:
        logger.error(f"Statement ingestion failed: {e}")
        return StatusResponse(status="error", message=str(e))
