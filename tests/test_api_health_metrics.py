from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from services.api.main import app

def test_health_and_ready():
    client = TestClient(app)
    h = client.get("/health").json()
    r = client.get("/ready").json()
    assert h["status"] == "ok"
    assert r["status"] in ("ready", "not_ready")

def test_metrics_endpoint():
    client = TestClient(app)
    m = client.get("/metrics")
    # 200 with at least one Prometheus metric line
    assert m.status_code == 200
    assert "http_requests_total" in m.text or "http_server_requests_seconds" in m.text or "process_cpu_seconds_total" in m.text
