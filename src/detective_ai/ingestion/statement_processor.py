"""Witness statement processor: chunking, embedding, and storage."""

from __future__ import annotations

import logging
from datetime import datetime

from detective_ai.core.enums import EvidenceType
from detective_ai.core.models import Evidence, WitnessStatement
from detective_ai.ingestion.embeddings import embed_text, embed_texts
from detective_ai.storage.database import db

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks for embedding.

    Args:
        text: Input text.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap between chunks.

    Returns:
        List of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at sentence boundary
        if end < len(text):
            for sep in [". ", ".\n", "\n\n", "\n", " "]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if c]


def process_statement(
    text: str,
    source: str,
    statement_time: datetime,
    event_time: datetime | None = None,
    reliability_score: float = 0.7,
    case_id: str = "",
) -> WitnessStatement:
    """Process a single witness statement: chunk, embed, and store.

    Args:
        text: The witness statement text.
        source: Witness identifier.
        statement_time: When the statement was given.
        event_time: When the described event occurred.
        reliability_score: Assessed reliability (0-1).
        case_id: Investigation case ID.

    Returns:
        WitnessStatement object.
    """
    statement = WitnessStatement(
        source=source,
        text=text,
        timestamp=statement_time,
        event_time=event_time,
        reliability_score=reliability_score,
        metadata={"case_id": case_id},
    )

    # Chunk and embed the statement
    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    # Use the first chunk's embedding as the main statement embedding
    main_embedding = embeddings[0] if embeddings else embed_text(text)

    # Create evidence record
    evidence = Evidence(
        id=statement.id,
        type=EvidenceType.WITNESS_STATEMENT,
        source=source,
        timestamp=statement_time,
        confidence_score=reliability_score,
        description=f"Witness statement from {source}: {text[:200]}...",
        metadata={
            "case_id": case_id,
            "event_time": event_time.isoformat() if event_time else None,
            "chunk_count": len(chunks),
        },
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
            embedding=main_embedding,
        )
        db.insert_statement(
            session,
            id=statement.id,
            source=statement.source,
            text=statement.text,
            timestamp=statement.timestamp,
            event_time=statement.event_time,
            reliability_score=statement.reliability_score,
            metadata_=statement.metadata,
            embedding=main_embedding,
        )

    logger.info(
        f"Processed statement from {source}: {len(chunks)} chunks, "
        f"reliability={reliability_score:.2f}"
    )
    return statement


def process_statements_batch(
    statements: list[dict],
    case_id: str = "",
) -> list[WitnessStatement]:
    """Process multiple witness statements.

    Args:
        statements: List of dicts with keys: text, source, timestamp,
                    event_time (optional), reliability_score (optional).
        case_id: Investigation case ID.

    Returns:
        List of WitnessStatement objects.
    """
    results = []
    for stmt_data in statements:
        timestamp = stmt_data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        event_time = stmt_data.get("event_time")
        if isinstance(event_time, str):
            event_time = datetime.fromisoformat(event_time)

        result = process_statement(
            text=stmt_data["text"],
            source=stmt_data.get("source", "anonymous"),
            statement_time=timestamp or datetime.utcnow(),
            event_time=event_time,
            reliability_score=stmt_data.get("reliability_score", 0.7),
            case_id=case_id,
        )
        results.append(result)

    return results
