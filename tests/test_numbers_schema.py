from __future__ import annotations
import json
import pathlib


def test_numbers_schema_complete():
    """Ensure numbers.json has all required top-level fields."""
    numbers_path = pathlib.Path("docs/numbers.json")
    if not numbers_path.exists():
        return  # No numbers to validate
    
    with open(numbers_path, 'r') as f:
        numbers = json.load(f)
    
    # Required top-level fields
    required_fields = [
        "metadata",
        "expected_panel", 
        "backtest",
        "constraints",
        "api",
        "explainability",
        "artifacts"
    ]
    
    for field in required_fields:
        assert field in numbers, f"numbers.json missing required field: {field}"
        assert numbers[field] is not None, f"numbers.json field '{field}' is null"


def test_backtest_metrics_present():
    """Ensure backtest section has required metrics."""
    numbers_path = pathlib.Path("docs/numbers.json")
    if not numbers_path.exists():
        return
    
    with open(numbers_path, 'r') as f:
        numbers = json.load(f)
    
    backtest = numbers.get("backtest", {})
    assert "constrained" in backtest, "Missing 'constrained' backtest metrics"
    
    constrained = backtest["constrained"]
    required_metrics = [
        "sharpe_annual",
        "vol_annual", 
        "cagr",
        "max_dd",
        "turnover_pct_day",
        "avg_gross",
        "hit_ratio",
        "trades_per_day"
    ]
    
    for metric in required_metrics:
        assert metric in constrained, f"Missing backtest metric: {metric}"
        assert isinstance(constrained[metric], (int, float)), f"Metric '{metric}' must be numeric"


def test_constraints_metrics_present():
    """Ensure constraints section has required metrics."""
    numbers_path = pathlib.Path("docs/numbers.json")
    if not numbers_path.exists():
        return
    
    with open(numbers_path, 'r') as f:
        numbers = json.load(f)
    
    constraints = numbers.get("constraints", {})
    required_metrics = [
        "max_net_exposure",
        "mean_net_exposure",
        "max_beta_deviation", 
        "mean_beta_deviation",
        "max_sector_l2",
        "mean_sector_l2",
        "infeasible_days",
        "infeasible_pct"
    ]
    
    for metric in required_metrics:
        assert metric in constraints, f"Missing constraint metric: {metric}"
        assert isinstance(constraints[metric], (int, float)), f"Constraint metric '{metric}' must be numeric"


def test_metadata_present():
    """Ensure metadata section has required provenance fields."""
    numbers_path = pathlib.Path("docs/numbers.json")
    if not numbers_path.exists():
        return
    
    with open(numbers_path, 'r') as f:
        numbers = json.load(f)
    
    metadata = numbers.get("metadata", {})
    required_fields = [
        "commit_sha",
        "timestamp",
        "python_version",
        "os"
    ]
    
    for field in required_fields:
        assert field in metadata, f"Missing metadata field: {field}"
        assert metadata[field], f"Metadata field '{field}' is empty"
