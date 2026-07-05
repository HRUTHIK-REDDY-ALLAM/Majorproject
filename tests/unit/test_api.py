"""API smoke tests using FastAPI TestClient over a temporary SQLite database."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Point the shared DatabaseManager singleton at a temp SQLite database
    from detective_ai.storage.database import db

    db.__init__(f"sqlite:///{(tmp_path / 'api_test.db').as_posix()}")
    db.init_db()

    # Keep tests offline: stub out the sentence-transformer embedder
    import detective_ai.ingestion.log_processor as log_processor

    monkeypatch.setattr(log_processor, "embed_text", lambda *a, **k: [0.0] * 384)

    from detective_ai.api.app import app

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestAPI:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_openapi_docs_exposed(self, client):
        assert client.get("/api/docs").status_code == 200

    def test_list_cases_empty(self, client):
        r = client.get("/api/v1/cases")
        assert r.status_code == 200
        assert r.json()["cases"] == []

    def test_ingest_access_logs(self, client):
        logs = [
            {
                "person_id": "EMP001", "person_name": "John Smith",
                "location": "Main Entrance",
                "timestamp": datetime(2025, 1, 15, 8, 5).isoformat(),
                "action": "entry",
            }
        ]
        r = client.post("/api/v1/ingest/logs", json={
            "content": json.dumps(logs), "format": "json",
            "source": "badge_system",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success", body["message"]
        assert body["data"]["entry_count"] == 1

    def test_report_for_unknown_case(self, client):
        r = client.get("/api/v1/report/nonexistent")
        assert r.status_code == 200
        assert r.json()["status"] == "error"

    def test_counterfactual_for_unknown_case(self, client):
        r = client.post("/api/v1/counterfactual/", json={
            "case_id": "nonexistent", "removed_evidence_id": "ev1",
        })
        assert r.status_code == 200
        assert r.json()["status"] == "error"
