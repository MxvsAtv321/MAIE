from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from fastapi.testclient import TestClient
from services.api.main import app

def test_score_expected_roundtrip(tmp_path, monkeypatch):
    # create a tiny expected_latest.parquet on the fly
    import pandas as pd
    p = tmp_path / "expected"; p.mkdir()
    df = pd.DataFrame([[0.01, -0.02]], index=pd.to_datetime(["2024-12-31"]), columns=["SIM0001","SIM0002"])
    (p / "expected_latest.parquet").write_bytes(df.to_parquet())

    # monkeypatch loader path inside the endpoint by chdir
    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        client = TestClient(app)
        resp = client.post("/score_expected", json={"tickers": ["SIM0001","SIM0002"]})
        j = resp.json()
        assert "alpha" in j and abs(j["alpha"]["SIM0001"] - 0.01) < 1e-9
    finally:
        os.chdir(cwd)
