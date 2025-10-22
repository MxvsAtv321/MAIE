"""
Test production thresholds.
"""
import pytest
import yaml
import json
from pathlib import Path

@pytest.mark.slow
def test_thresholds_pass():
    """Test that all production thresholds are met."""
    # Load thresholds
    thresholds_file = Path("audit_thresholds.yaml")
    if not thresholds_file.exists():
        pytest.skip("audit_thresholds.yaml not found")
    
    with open(thresholds_file) as f:
        thresholds = yaml.safe_load(f)
    
    # Load numbers
    numbers_file = Path("docs/numbers.json")
    if not numbers_file.exists():
        pytest.skip("docs/numbers.json not found - run 'make audit' first")
    
    with open(numbers_file) as f:
        numbers = json.load(f)
    
    # Check each threshold
    violations = []
    
    # Risk & performance checks
    backtest = numbers.get("backtest", {})
    if "sharpe_ann" in backtest:
        sharpe = backtest["sharpe_ann"]
        min_sharpe = thresholds.get("sharpe_min", -0.10)
        if sharpe < min_sharpe:
            violations.append(f"Sharpe {sharpe:.3f} < {min_sharpe}")
    
    if "vol_ann" in backtest:
        vol = backtest["vol_ann"]
        max_vol = thresholds.get("vol_ann_max", 0.50)
        if vol > max_vol:
            violations.append(f"Volatility {vol:.3f} > {max_vol}")
    
    if "max_dd" in backtest:
        maxdd = backtest["max_dd"]
        max_dd_limit = thresholds.get("maxdd_max", 0.20)
        if maxdd > max_dd_limit:
            violations.append(f"Max DD {maxdd:.3f} > {max_dd_limit}")
    
    if "turnover_daily" in backtest:
        turnover = backtest["turnover_daily"]
        max_turnover = thresholds.get("turnover_daily_max", 0.15)
        if turnover > max_turnover:
            violations.append(f"Daily turnover {turnover:.3f} > {max_turnover}")
    
    if "hit_ratio" in backtest:
        hit_ratio = backtest["hit_ratio"]
        min_hit = thresholds.get("hit_ratio_min", 0.45)
        if hit_ratio < min_hit:
            violations.append(f"Hit ratio {hit_ratio:.3f} < {min_hit}")
    
    # Constraint checks
    constraints = numbers.get("constraints", {})
    if "beta_residual_p95" in constraints:
        beta_res = constraints["beta_residual_p95"]
        beta_tol = thresholds.get("beta_tol", 0.001)
        if beta_res > beta_tol:
            violations.append(f"Beta residual P95 {beta_res:.6f} > {beta_tol}")
    
    if "sector_residual_p95" in constraints:
        sector_res = constraints["sector_residual_p95"]
        sector_tol = thresholds.get("sector_tol", 0.0005)
        if sector_res > sector_tol:
            violations.append(f"Sector residual P95 {sector_res:.6f} > {sector_tol}")
    
    # QP infeasibility check
    backtest_meta = numbers.get("backtest_meta", {})
    if "infeasible_days" in backtest_meta and "n_days" in backtest_meta:
        infeasible_ratio = backtest_meta["infeasible_days"] / max(1, backtest_meta["n_days"])
        max_infeasible = thresholds.get("qp_infeasible_ratio_max", 0.001)
        if infeasible_ratio > max_infeasible:
            violations.append(f"QP infeasible ratio {infeasible_ratio:.4f} > {max_infeasible}")
    
    # API latency checks
    api = numbers.get("api", {})
    score_expected = api.get("score_expected", {})
    if "p95_ms" in score_expected:
        p95_ms = score_expected["p95_ms"]
        max_p95 = thresholds.get("score_expected_p95_ms", 200)
        if p95_ms > max_p95:
            violations.append(f"/score_expected P95 {p95_ms:.1f}ms > {max_p95}ms")
    
    explain_local = api.get("explain_local", {})
    if "p95_ms" in explain_local:
        p95_ms = explain_local["p95_ms"]
        max_p95 = thresholds.get("explain_local_p95_ms", 400)
        if p95_ms > max_p95:
            violations.append(f"/explain_local P95 {p95_ms:.1f}ms > {max_p95}ms")
    
    # Assert no violations
    if violations:
        pytest.fail(f"Threshold violations: {'; '.join(violations)}")
    
    # If we get here, all thresholds passed
    assert True
