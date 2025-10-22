from __future__ import annotations
import json
import pathlib


def test_realistic_backtest_metrics_nonzero_when_csvs_present():
    """Ensure realistic backtest metrics are computed from actual CSV data, not placeholders."""
    # If outputs_from_expected has returns/weights, forbid zeros from placeholders
    outdir = pathlib.Path("outputs_from_expected")
    returns = list(outdir.glob("returns_*.csv"))
    weights = list(outdir.glob("weights_*.csv"))
    
    if not (returns and weights):
        return  # nothing to assert

    numbers_path = pathlib.Path("docs/numbers.json")
    if not numbers_path.exists():
        return  # no numbers to validate
    
    numbers = json.loads(numbers_path.read_text())
    cons = numbers.get("backtest", {}).get("constrained", {})
    
    # Only assert "is present & computed"; not the exact values (keep synthetic tolerant)
    for k in ["turnover_pct_day", "avg_gross", "hit_ratio", "trades_per_day"]:
        assert k in cons, f"missing metric: {k}"
        assert cons[k] != 0.0, f"{k} equals 0.0 — likely a placeholder, not computed"


def test_expected_panel_metadata_consistency():
    """Ensure expected panel metadata is consistent between sources."""
    numbers_path = pathlib.Path("docs/numbers.json")
    expected_meta_path = pathlib.Path("expected/metadata.json")
    
    if not (numbers_path.exists() and expected_meta_path.exists()):
        return  # no data to validate
    
    numbers = json.loads(numbers_path.read_text())
    expected_meta = json.loads(expected_meta_path.read_text())
    
    # Check that numbers.json uses expected_meta as source of truth
    expected_panel = numbers.get("expected_panel", {})
    
    # Shape should match
    if expected_meta.get("shape") and expected_panel.get("shape"):
        assert expected_panel["shape"] == expected_meta["shape"], \
            f"Shape mismatch: numbers.json {expected_panel['shape']} vs metadata.json {expected_meta['shape']}"
    
    # File count should match
    if expected_meta.get("n_files") and expected_panel.get("n_files"):
        assert expected_panel["n_files"] == expected_meta["n_files"], \
            f"File count mismatch: numbers.json {expected_panel['n_files']} vs metadata.json {expected_meta['n_files']}"


def test_constraint_residuals_present_when_backtest_exists():
    """Ensure constraint residuals are computed when backtest outputs exist."""
    outdir = pathlib.Path("outputs_from_expected")
    cutout_files = list(outdir.glob("cutout_*.csv"))
    
    if not cutout_files:
        return  # no constraint data to validate
    
    numbers_path = pathlib.Path("docs/numbers.json")
    if not numbers_path.exists():
        return
    
    numbers = json.loads(numbers_path.read_text())
    constraints = numbers.get("constraints", {})
    
    # Should have constraint metrics if cutout files exist
    expected_fields = ["max_net_exposure", "mean_net_exposure", "max_beta_deviation", 
                       "mean_beta_deviation", "max_sector_l2", "mean_sector_l2"]
    
    for field in expected_fields:
        assert field in constraints, f"Missing constraint metric: {field}"
        # Don't assert non-zero here as synthetic data might legitimately have zero residuals
