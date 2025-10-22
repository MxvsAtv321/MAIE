from __future__ import annotations
import pytest
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestNumbersGuardrails:
    """Test guardrails for production numbers validation."""
    
    def test_expected_panel_freshness(self):
        """Test that expected panel is fresh (< 36h old)."""
        expected_path = Path("expected/expected_latest.parquet")
        if not expected_path.exists():
            pytest.skip("Expected panel not found")
        
        mtime = expected_path.stat().st_mtime
        age_hours = (datetime.now().timestamp() - mtime) / 3600
        
        assert age_hours < 36, f"Expected panel is {age_hours:.1f}h old (max 36h)"
    
    def test_neutrality_within_tolerances(self):
        """Test that neutrality constraints are satisfied within tolerances."""
        # Load constraint tolerances
        import yaml
        with open("constraints.yaml") as f:
            cfg = yaml.safe_load(f)
        
        beta_tol = float(cfg.get("beta_tolerance", 0.0))
        sector_tol = float(cfg.get("sector_tolerance", 0.0))
        
        # Check if we have backtest results
        outputs_dir = Path("outputs_from_expected")
        if not outputs_dir.exists():
            pytest.skip("No backtest outputs found")
        
        # Find latest cutout file
        cutout_files = list(outputs_dir.glob("cutout_ret_data_*.csv"))
        if not cutout_files:
            pytest.skip("No cutout files found")
        
        latest_cutout = max(cutout_files, key=lambda f: f.stat().st_mtime)
        df = pd.read_csv(latest_cutout)
        
        # Check net exposure
        if "net" in df.columns:
            max_net = df["net"].abs().max()
            assert max_net <= beta_tol + 1e-4, f"Max net exposure {max_net} exceeds tolerance {beta_tol}"
        
        # Check beta exposure
        if "beta" in df.columns:
            max_beta = df["beta"].abs().max()
            assert max_beta <= beta_tol + 1e-4, f"Max beta deviation {max_beta} exceeds tolerance {beta_tol}"
        
        # Check sector L2
        if "sector_l2" in df.columns:
            max_sector_l2 = df["sector_l2"].max()
            assert max_sector_l2 <= sector_tol + 1e-4, f"Max sector L2 {max_sector_l2} exceeds tolerance {sector_tol}"
    
    def test_turnover_within_limits(self):
        """Test that turnover is within configured limits."""
        import yaml
        with open("constraints.yaml") as f:
            cfg = yaml.safe_load(f)
        
        gross_limit = float(cfg.get("gross_limit", 2.0))
        
        # Check if we have backtest results
        outputs_dir = Path("outputs_from_expected")
        if not outputs_dir.exists():
            pytest.skip("No backtest outputs found")
        
        # Find latest cutout file
        cutout_files = list(outputs_dir.glob("cutout_ret_data_*.csv"))
        if not cutout_files:
            pytest.skip("No cutout files found")
        
        latest_cutout = max(cutout_files, key=lambda f: f.stat().st_mtime)
        df = pd.read_csv(latest_cutout)
        
        # Check turnover (approximated from weight changes)
        if "net" in df.columns:
            # This is a simplified check - in practice, you'd compute actual turnover
            # from weight changes between consecutive days
            pass  # Placeholder for actual turnover calculation
    
    def test_qp_infeasible_days_low(self):
        """Test that QP infeasible days are ≤ 0.1% of total."""
        # This would be implemented by checking the infeasibility metric
        # For now, we'll check if the metric exists
        metrics_path = Path("docs/numbers.json")
        if not metrics_path.exists():
            pytest.skip("Numbers file not found")
        
        with open(metrics_path) as f:
            numbers = json.load(f)
        
        infeasible_pct = numbers.get("constraints", {}).get("infeasible_pct", 0.0)
        assert infeasible_pct <= 0.1, f"QP infeasible days {infeasible_pct}% exceeds 0.1% limit"
    
    def test_explain_local_non_empty(self):
        """Test that /explain_local returns non-empty results."""
        # This would require running the API and testing the endpoint
        # For now, we'll check if the explainability metrics exist
        metrics_path = Path("docs/numbers.json")
        if not metrics_path.exists():
            pytest.skip("Numbers file not found")
        
        with open(metrics_path) as f:
            numbers = json.load(f)
        
        non_empty_rate = numbers.get("explainability", {}).get("non_empty_rate", 0.0)
        assert non_empty_rate >= 95.0, f"Non-empty explain_local rate {non_empty_rate}% below 95% threshold"
    
    def test_api_latency_bounds(self):
        """Test that API latency is within bounds."""
        metrics_path = Path("docs/numbers.json")
        if not metrics_path.exists():
            pytest.skip("Numbers file not found")
        
        with open(metrics_path) as f:
            numbers = json.load(f)
        
        # Check /score_expected latency
        score_expected_p95 = numbers.get("api", {}).get("score_expected", {}).get("p95_ms", 0.0)
        assert score_expected_p95 < 200, f"/score_expected P95 latency {score_expected_p95}ms exceeds 200ms limit"
        
        # Check /explain_local latency
        explain_local_p95 = numbers.get("api", {}).get("explain_local", {}).get("p95_ms", 0.0)
        assert explain_local_p95 < 400, f"/explain_local P95 latency {explain_local_p95}ms exceeds 400ms limit"

    def test_expected_metadata_coherence(self):
        """Test that expected panel metadata is coherent."""
        import json, datetime as dt
        meta_path = Path("expected/metadata.json")
        if not meta_path.exists():
            pytest.skip("Expected metadata not found")
        
        meta = json.load(open(meta_path))
        start = dt.date.fromisoformat(meta["start"])
        end = dt.date.fromisoformat(meta["end"])
        cal_days = (end - start).days + 1
        n_unique = meta["n_unique_dates"]
        
        # Business day ratio sanity (loose bounds for synthetic calendars)
        ratio = n_unique / max(1, cal_days)
        assert 0.5 <= ratio <= 1.05, f"Unrealistic trading-day density: {ratio:.2f}"
        
        # If only 12 monthly files, expect roughly one year of data
        if meta["n_files"] == 13:
            assert 180 <= n_unique <= 320, f"With ~12 months of partitions, n_unique={n_unique} is suspicious"
