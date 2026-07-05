"""pgvector similarity search operations.

Provides high-level functions for vector similarity search
across evidence and visual detection embeddings stored in PostgreSQL.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def similarity_search(
    session: Session,
    query_embedding: list[float],
    table: str = "evidence",
    embedding_column: str = "embedding",
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    """Perform cosine similarity search on a pgvector-enabled table.

    Args:
        session: SQLAlchemy session.
        query_embedding: Query vector.
        table: Table name to search.
        embedding_column: Column containing the vector.
        top_k: Maximum number of results.
        filters: Optional column=value filters.
        min_similarity: Minimum cosine similarity threshold (0-1).

    Returns:
        List of dicts with row data and similarity score.
    """
    if session.get_bind().dialect.name != "postgresql":
        return _similarity_search_python(
            session, query_embedding, table, embedding_column,
            top_k, filters, min_similarity,
        )

    # Build the base query with cosine distance operator (<=>)
    embedding_str = str(query_embedding)
    where_clauses = []
    params: dict[str, Any] = {"top_k": top_k}

    if filters:
        for i, (col, val) in enumerate(filters.items()):
            param_name = f"filter_{i}"
            where_clauses.append(f"{col} = :{param_name}")
            params[param_name] = val

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    query = text(f"""
        SELECT *,
               1 - ({embedding_column} <=> '{embedding_str}'::vector) AS similarity
        FROM {table}
        {where_sql}
        ORDER BY {embedding_column} <=> '{embedding_str}'::vector
        LIMIT :top_k
    """)

    result = session.execute(query, params)
    rows = []
    for row in result.mappings():
        row_dict = dict(row)
        if row_dict.get("similarity", 0) >= min_similarity:
            rows.append(row_dict)

    logger.debug(f"Vector search on {table}: {len(rows)} results (top_k={top_k})")
    return rows


def _similarity_search_python(
    session: Session,
    query_embedding: list[float],
    table: str,
    embedding_column: str,
    top_k: int,
    filters: dict[str, Any] | None,
    min_similarity: float,
) -> list[dict[str, Any]]:
    """Cosine similarity computed in Python for non-pgvector databases (SQLite).

    Embeddings are stored as JSON text on these dialects, so we fetch the
    candidate rows and rank them with numpy. Fine for dev-scale data.
    """
    import json as _json

    where_clauses = [f"{embedding_column} IS NOT NULL"]
    params: dict[str, Any] = {}
    if filters:
        for i, (col, val) in enumerate(filters.items()):
            param_name = f"filter_{i}"
            where_clauses.append(f"{col} = :{param_name}")
            params[param_name] = val

    query = text(f"SELECT * FROM {table} WHERE {' AND '.join(where_clauses)}")
    q = np.array(query_embedding, dtype=float)
    q_norm = np.linalg.norm(q)

    scored: list[dict[str, Any]] = []
    for row in session.execute(query, params).mappings():
        row_dict = dict(row)
        raw = row_dict.get(embedding_column)
        if raw is None:
            continue
        vec = np.array(_json.loads(raw) if isinstance(raw, str) else raw, dtype=float)
        denom = q_norm * np.linalg.norm(vec)
        similarity = float(np.dot(q, vec) / denom) if denom else 0.0
        if similarity >= min_similarity:
            row_dict["similarity"] = similarity
            scored.append(row_dict)

    scored.sort(key=lambda r: r["similarity"], reverse=True)
    logger.debug(f"Python vector search on {table}: {len(scored[:top_k])} results")
    return scored[:top_k]


def search_evidence_embeddings(
    session: Session,
    query_embedding: list[float],
    evidence_type: str | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search evidence table by embedding similarity."""
    filters = {}
    if evidence_type:
        filters["type"] = evidence_type
    return similarity_search(
        session, query_embedding, table="evidence",
        embedding_column="embedding", top_k=top_k, filters=filters,
    )


def search_appearance_embeddings(
    session: Session,
    query_embedding: list[float],
    camera_id: str | None = None,
    top_k: int = 10,
    min_similarity: float = 0.6,
) -> list[dict[str, Any]]:
    """Search visual detections by appearance embedding similarity (ReID)."""
    filters = {}
    if camera_id:
        filters["camera_id"] = camera_id
    return similarity_search(
        session, query_embedding, table="visual_detections",
        embedding_column="appearance_embedding", top_k=top_k,
        filters=filters, min_similarity=min_similarity,
    )


def search_statement_embeddings(
    session: Session,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Search witness statements by text embedding similarity."""
    return similarity_search(
        session, query_embedding, table="witness_statements",
        embedding_column="embedding", top_k=top_k,
    )


def batch_insert_embeddings(
    session: Session,
    table: str,
    rows: list[dict[str, Any]],
) -> int:
    """Bulk insert rows with embeddings into a pgvector table.

    Args:
        session: SQLAlchemy session.
        table: Target table name.
        rows: List of dicts, each containing column values including an embedding.

    Returns:
        Number of rows inserted.
    """
    if not rows:
        return 0

    columns = list(rows[0].keys())
    col_str = ", ".join(columns)
    val_placeholders = ", ".join(f":{c}" for c in columns)

    query = text(f"INSERT INTO {table} ({col_str}) VALUES ({val_placeholders})")

    for row in rows:
        # Convert numpy arrays to lists for pgvector
        for k, v in row.items():
            if isinstance(v, np.ndarray):
                row[k] = v.tolist()

    session.execute(query, rows)
    logger.info(f"Batch inserted {len(rows)} rows into {table}")
    return len(rows)
