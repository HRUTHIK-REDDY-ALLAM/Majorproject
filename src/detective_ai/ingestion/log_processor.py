"""Access-log processor: parsing, embedding, and storage."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from io import StringIO
from typing import Any

from detective_ai.core.enums import EvidenceType
from detective_ai.core.models import AccessLogEntry, Evidence
from detective_ai.ingestion.embeddings import embed_text
from detective_ai.storage.database import db

logger = logging.getLogger(__name__)


def parse_csv_logs(csv_content: str) -> list[dict[str, Any]]:
    """Parse CSV access log data.

    Expected columns: person_id, person_name, location, timestamp, action
    """
    reader = csv.DictReader(StringIO(csv_content))
    records = []
    for row in reader:
        records.append({
            "person_id": row.get("person_id", ""),
            "person_name": row.get("person_name", ""),
            "location": row.get("location", ""),
            "timestamp": row.get("timestamp", ""),
            "action": row.get("action", "entry"),
        })
    return records


def parse_json_logs(json_content: str) -> list[dict[str, Any]]:
    """Parse JSON access log data."""
    data = json.loads(json_content)
    if isinstance(data, dict):
        data = data.get("logs", data.get("entries", [data]))
    return data


def process_access_logs(
    content: str,
    format: str = "json",
    source: str = "badge_system",
    case_id: str = "",
) -> list[AccessLogEntry]:
    """Process access log data: parse, embed, and store.

    Args:
        content: Raw log content (CSV or JSON string).
        format: "csv" or "json".
        source: Source system identifier.
        case_id: Investigation case ID.

    Returns:
        List of AccessLogEntry objects created.
    """
    if format == "csv":
        raw_records = parse_csv_logs(content)
    else:
        raw_records = parse_json_logs(content)

    entries = []

    for record in raw_records:
        # Parse timestamp
        ts_str = record.get("timestamp", "")
        try:
            timestamp = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            logger.warning(f"Skipping log with invalid timestamp: {ts_str}")
            continue

        entry = AccessLogEntry(
            person_id=record.get("person_id", "unknown"),
            person_name=record.get("person_name", ""),
            location=record.get("location", "unknown"),
            timestamp=timestamp,
            action=record.get("action", "entry"),
            metadata={"case_id": case_id, "raw": record},
        )

        # Create an evidence record with text embedding
        description = (
            f"{entry.person_name or entry.person_id} {entry.action} at "
            f"{entry.location} on {timestamp.isoformat()}"
        )
        embedding = embed_text(description)

        evidence = Evidence(
            id=entry.id,
            type=EvidenceType.ACCESS_LOG,
            source=source,
            timestamp=timestamp,
            confidence_score=0.95,  # badge logs are high-confidence
            description=description,
            metadata=entry.metadata,
        )

        # Store in database
        with db.session() as session:
            db.insert_evidence(
                session,
                id=evidence.id,
                type=evidence.type.value,
                source=evidence.source,
                timestamp=evidence.timestamp,
                confidence_score=evidence.confidence_score,
                description=evidence.description,
                metadata_=evidence.metadata,
                embedding=embedding,
            )
            db.insert_access_log(
                session,
                id=entry.id,
                person_id=entry.person_id,
                person_name=entry.person_name,
                location=entry.location,
                timestamp=entry.timestamp,
                action=entry.action,
                metadata_=entry.metadata,
            )

        entries.append(entry)

    logger.info(f"Processed {len(entries)} access log entries from {source}")
    return entries
