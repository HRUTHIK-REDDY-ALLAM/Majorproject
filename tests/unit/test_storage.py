"""Tests for the portable storage layer (SQLite dialect)."""

from __future__ import annotations

from datetime import datetime

import pytest

from detective_ai.storage import vector_ops
from detective_ai.storage.database import DatabaseManager


@pytest.fixture()
def db(tmp_path):
    manager = DatabaseManager(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    manager.init_db()
    return manager


class TestSQLiteStorage:
    def test_init_creates_tables(self, db):
        with db.session() as s:
            assert db.list_cases(s) == []

    def test_evidence_roundtrip_with_embedding(self, db):
        embedding = [0.5] * 384
        with db.session() as s:
            db.insert_evidence(
                s, id="ev1", type="access_log", source="badge",
                timestamp=datetime(2025, 1, 15, 9, 0), confidence_score=0.9,
                description="badge swipe", metadata_={"door": "main"},
                embedding=embedding,
            )
        with db.session() as s:
            row = db.get_evidence(s, "ev1")
            assert row is not None
            assert row.metadata_ == {"door": "main"}
            assert len(row.embedding) == 384
            assert row.embedding[0] == pytest.approx(0.5)

    def test_case_lifecycle(self, db):
        with db.session() as s:
            db.insert_case(s, id="case1", title="Test Case", max_rounds=3)
        with db.session() as s:
            db.update_case(s, "case1", status="completed",
                           report_data={"summary": "done"})
        with db.session() as s:
            case = db.get_case(s, "case1")
            assert case.status == "completed"
            assert case.report_data["summary"] == "done"

    def test_hypothesis_crud(self, db):
        with db.session() as s:
            db.insert_hypothesis(
                s, id="h1", case_id="case1", title="Suspect X did it",
                confidence=0.7, supporting_evidence=[{"evidence_id": "ev1"}],
            )
            db.update_hypothesis(s, "h1", status="pruned",
                                 rejection_reason="alibi confirmed")
        with db.session() as s:
            rows = db.get_hypotheses(s, "case1")
            assert len(rows) == 1
            assert rows[0].status == "pruned"
            assert rows[0].supporting_evidence == [{"evidence_id": "ev1"}]

    def test_vector_search_python_fallback(self, db):
        with db.session() as s:
            db.insert_evidence(
                s, id="near", type="witness_statement", source="w1",
                timestamp=datetime(2025, 1, 15), confidence_score=0.8,
                embedding=[1.0] + [0.0] * 383,
            )
            db.insert_evidence(
                s, id="far", type="witness_statement", source="w2",
                timestamp=datetime(2025, 1, 15), confidence_score=0.8,
                embedding=[0.0, 1.0] + [0.0] * 382,
            )
        with db.session() as s:
            results = vector_ops.search_evidence_embeddings(
                s, [1.0] + [0.0] * 383, top_k=2,
            )
            assert results[0]["id"] == "near"
            assert results[0]["similarity"] == pytest.approx(1.0)
            assert results[1]["similarity"] < 0.5

    def test_access_log_filters(self, db):
        with db.session() as s:
            db.insert_access_log(
                s, id="a1", person_id="EMP001", person_name="John",
                location="Server Room", timestamp=datetime(2025, 1, 15, 9, 30),
            )
            db.insert_access_log(
                s, id="a2", person_id="EMP002", person_name="Jane",
                location="Lobby", timestamp=datetime(2025, 1, 15, 10, 0),
            )
        with db.session() as s:
            logs = db.get_access_logs(s, person_id="EMP001")
            assert [log.id for log in logs] == ["a1"]
            logs = db.get_access_logs(
                s, start_time=datetime(2025, 1, 15, 9, 45),
            )
            assert [log.id for log in logs] == ["a2"]
