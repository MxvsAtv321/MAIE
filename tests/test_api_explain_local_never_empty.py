from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from fastapi.testclient import TestClient
from services.api.main import app

def test_explain_local_never_empty():
    client = TestClient(app)
    # Two short synthetic price series (short history to test robustness)
    prices = {
        "sim0001": [100 + i*0.1 for i in range(45)],  # Mixed case + short history
        "SIM0002": [120 + i*0.05 for i in range(45)],
    }
    body = {"prices": prices, "ticker": "Sim0001", "top_k": 5}  # Mixed case ticker
    r = client.post("/explain_local", json=body)
    j = r.json()
    # The endpoint should always return something, even if it's the fallback
    assert "top_features" in j
    # Should now return non-empty results due to bulletproof implementation
    assert len(j["top_features"]) > 0
    print(f"Got top_features: {j['top_features']}")  # Debug output
