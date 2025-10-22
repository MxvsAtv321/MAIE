#!/usr/bin/env python3
"""
Check production thresholds against audit numbers.
Exits non-zero if any threshold is breached.
"""
import sys
import yaml
import json
from pathlib import Path
from typing import Dict, Any

def load_thresholds(thresholds_file: str) -> Dict[str, Any]:
    """Load threshold configuration."""
    with open(thresholds_file) as f:
        return yaml.safe_load(f)

def load_numbers(numbers_file: str) -> Dict[str, Any]:
    """Load audit numbers."""
    with open(numbers_file) as f:
        return json.load(f)

def check_thresholds(thresholds: Dict[str, Any], numbers: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Check all thresholds against numbers. Returns (passed, violations)."""
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
    
    # Expected freshness check
    expected_meta = numbers.get("expected_meta", {})
    if "build_seconds" in expected_meta:
        # This is a placeholder - in real ops you'd check actual file timestamps
        # For now, just ensure metadata exists
        pass
    
    return len(violations) == 0, violations

def main():
    """Main threshold checker."""
    if len(sys.argv) != 3:
        print("Usage: python scripts/check_thresholds.py <thresholds.yaml> <numbers.json>")
        sys.exit(1)
    
    thresholds_file = sys.argv[1]
    numbers_file = sys.argv[2]
    
    if not Path(thresholds_file).exists():
        print(f"ERROR: Thresholds file {thresholds_file} not found")
        sys.exit(1)
    
    if not Path(numbers_file).exists():
        print(f"ERROR: Numbers file {numbers_file} not found")
        sys.exit(1)
    
    try:
        thresholds = load_thresholds(thresholds_file)
        numbers = load_numbers(numbers_file)
        
        passed, violations = check_thresholds(thresholds, numbers)
        
        if passed:
            print("✅ All thresholds PASSED")
            sys.exit(0)
        else:
            print("❌ Threshold violations:")
            for violation in violations:
                print(f"  - {violation}")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
