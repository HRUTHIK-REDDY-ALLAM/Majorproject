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
    camera_id: str = Form(...),
    start_time: str = Form(...),
    case_id: str = Form(""),
):
    """Upload and process a video file for evidence extraction."""
    from datetime import datetime

    from detective_ai.ingestion.video_processor import process_video

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
            start_time=datetime.fromisoformat(start_time),
            case_id=case_id,
        )
        return StatusResponse(
            status="success",
            message=f"Processed {len(evidence_items)} evidence items from video",
            data={"evidence_count": len(evidence_items), "camera_id": camera_id},
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
    try:
        statement = process_statement(
            text=request.text,
            source=request.source,
            statement_time=request.timestamp,
            event_time=request.event_time,
            reliability_score=request.reliability_score,
            case_id=request.case_id,
        )
        return StatusResponse(
            status="success",
            message=f"Processed statement from {request.source}",
            data={"statement_id": statement.id},
        )
    except Exception as e:
        logger.error(f"Statement ingestion failed: {e}")
        return StatusResponse(status="error", message=str(e))
