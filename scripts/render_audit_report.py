#!/usr/bin/env python3
"""Render audit report from numbers.json."""

from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def get_commit_sha() -> str:
    """Get current commit SHA."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "unknown"


def get_timestamp() -> str:
    """Get current timestamp."""
    return datetime.utcnow().isoformat() + "Z"


def get_python_version() -> str:
    """Get Python version."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_os_info() -> str:
    """Get OS information."""
    import os
    return f"{os.uname().sysname} {os.uname().machine}"


def render_audit_report():
    """Render the audit report from numbers.json."""
    numbers_path = Path("docs/numbers.json")
    if not numbers_path.exists():
        print("Error: docs/numbers.json not found. Run 'make audit' first.")
        return
    
    with open(numbers_path) as f:
        numbers = json.load(f)
    
    # Check for dirty git tree
    import subprocess
    try:
        dirty_output = subprocess.run(["git", "status", "--porcelain"], 
                                    capture_output=True, text=True).stdout.strip()
        dirty_tree = dirty_output != ""
    except Exception:
        dirty_tree = False
    
    # Read the template
    template_path = Path("docs/audit_report.md")
    if not template_path.exists():
        print("Error: docs/audit_report.md template not found.")
        return
    
    with open(template_path) as f:
        template = template.read()
    
    # Replace placeholders with actual values
    template = template.replace("$(date)", get_timestamp())
    template = template.replace("$(git rev-parse HEAD)", get_commit_sha())
    template = template.replace("$(python --version)", f"Python {get_python_version()}")
    template = template.replace("$(uname -s) $(uname -m)", get_os_info())
    
    # Replace JSON values
    template = template.replace("$(jq -r '.expected_panel.shape' docs/numbers.json)", str(numbers.get("expected_panel", {}).get("shape", [0, 0])))
    template = template.replace("$(jq -r '.expected_panel.start_date' docs/numbers.json)", str(numbers.get("expected_panel", {}).get("start_date", "")))
    template = template.replace("$(jq -r '.expected_panel.end_date' docs/numbers.json)", str(numbers.get("expected_panel", {}).get("end_date", "")))
    template = template.replace("$(jq -r '.expected_panel.file_count' docs/numbers.json)", str(numbers.get("expected_panel", {}).get("file_count", 0)))
    template = template.replace("$(jq -r '.expected_panel.total_bytes' docs/numbers.json)", str(numbers.get("expected_panel", {}).get("total_bytes", 0)))
    template = template.replace("$(jq -r '.expected_panel.build_time_seconds' docs/numbers.json)", str(numbers.get("expected_panel", {}).get("build_time_seconds", 0.0)))
    
    # Backtest metrics
    unconstrained = numbers.get("backtest", {}).get("unconstrained", {})
    constrained = numbers.get("backtest", {}).get("constrained", {})
    
    template = template.replace("$(jq -r '.backtest.unconstrained.sharpe_annual' docs/numbers.json)", str(unconstrained.get("sharpe_annual", 0.0)))
    template = template.replace("$(jq -r '.backtest.unconstrained.vol_annual' docs/numbers.json)", str(unconstrained.get("vol_annual", 0.0)))
    template = template.replace("$(jq -r '.backtest.unconstrained.cagr' docs/numbers.json)", str(unconstrained.get("cagr", 0.0)))
    template = template.replace("$(jq -r '.backtest.unconstrained.max_dd' docs/numbers.json)", str(unconstrained.get("max_dd", 0.0)))
    template = template.replace("$(jq -r '.backtest.unconstrained.turnover_pct_day' docs/numbers.json)", str(unconstrained.get("turnover_pct_day", 0.0)))
    template = template.replace("$(jq -r '.backtest.unconstrained.avg_gross' docs/numbers.json)", str(unconstrained.get("avg_gross", 0.0)))
    template = template.replace("$(jq -r '.backtest.unconstrained.hit_ratio' docs/numbers.json)", str(unconstrained.get("hit_ratio", 0.0)))
    template = template.replace("$(jq -r '.backtest.unconstrained.trades_per_day' docs/numbers.json)", str(unconstrained.get("trades_per_day", 0.0)))
    
    template = template.replace("$(jq -r '.backtest.constrained.sharpe_annual' docs/numbers.json)", str(constrained.get("sharpe_annual", 0.0)))
    template = template.replace("$(jq -r '.backtest.constrained.vol_annual' docs/numbers.json)", str(constrained.get("vol_annual", 0.0)))
    template = template.replace("$(jq -r '.backtest.constrained.cagr' docs/numbers.json)", str(constrained.get("cagr", 0.0)))
    template = template.replace("$(jq -r '.backtest.constrained.max_dd' docs/numbers.json)", str(constrained.get("max_dd", 0.0)))
    template = template.replace("$(jq -r '.backtest.constrained.turnover_pct_day' docs/numbers.json)", str(constrained.get("turnover_pct_day", 0.0)))
    template = template.replace("$(jq -r '.backtest.constrained.avg_gross' docs/numbers.json)", str(constrained.get("avg_gross", 0.0)))
    template = template.replace("$(jq -r '.backtest.constrained.hit_ratio' docs/numbers.json)", str(constrained.get("hit_ratio", 0.0)))
    template = template.replace("$(jq -r '.backtest.constrained.trades_per_day' docs/numbers.json)", str(constrained.get("trades_per_day", 0.0)))
    
    # Constraint residuals
    constraints = numbers.get("constraints", {})
    template = template.replace("$(jq -r '.constraints.max_net_exposure' docs/numbers.json)", str(constraints.get("max_net_exposure", 0.0)))
    template = template.replace("$(jq -r '.constraints.mean_net_exposure' docs/numbers.json)", str(constraints.get("mean_net_exposure", 0.0)))
    template = template.replace("$(jq -r '.constraints.max_beta_deviation' docs/numbers.json)", str(constraints.get("max_beta_deviation", 0.0)))
    template = template.replace("$(jq -r '.constraints.mean_beta_deviation' docs/numbers.json)", str(constraints.get("mean_beta_deviation", 0.0)))
    template = template.replace("$(jq -r '.constraints.max_sector_l2' docs/numbers.json)", str(constraints.get("max_sector_l2", 0.0)))
    template = template.replace("$(jq -r '.constraints.mean_sector_l2' docs/numbers.json)", str(constraints.get("mean_sector_l2", 0.0)))
    template = template.replace("$(jq -r '.constraints.infeasible_days' docs/numbers.json)", str(constraints.get("infeasible_days", 0)))
    template = template.replace("$(jq -r '.constraints.infeasible_pct' docs/numbers.json)", str(constraints.get("infeasible_pct", 0.0)))
    
    # API performance
    api = numbers.get("api", {})
    score_expected = api.get("score_expected", {})
    explain_local = api.get("explain_local", {})
    
    template = template.replace("$(jq -r '.api.score_expected.median_ms' docs/numbers.json)", str(score_expected.get("median_ms", 0.0)))
    template = template.replace("$(jq -r '.api.score_expected.p95_ms' docs/numbers.json)", str(score_expected.get("p95_ms", 0.0)))
    template = template.replace("$(jq -r '.api.score_expected.error_rate' docs/numbers.json)", str(score_expected.get("error_rate", 0.0)))
    template = template.replace("$(jq -r '.api.explain_local.median_ms' docs/numbers.json)", str(explain_local.get("median_ms", 0.0)))
    template = template.replace("$(jq -r '.api.explain_local.p95_ms' docs/numbers.json)", str(explain_local.get("p95_ms", 0.0)))
    template = template.replace("$(jq -r '.api.explain_local.error_rate' docs/numbers.json)", str(explain_local.get("error_rate", 0.0)))
    
    # Explainability
    explainability = numbers.get("explainability", {})
    template = template.replace("$(jq -r '.explainability.non_empty_rate' docs/numbers.json)", str(explainability.get("non_empty_rate", 0.0)))
    template = template.replace("$(jq -r '.explainability.pred_contrib_rate' docs/numbers.json)", str(explainability.get("pred_contrib_rate", 0.0)))
    template = template.replace("$(jq -r '.explainability.tree_explainer_rate' docs/numbers.json)", str(explainability.get("tree_explainer_rate", 0.0)))
    template = template.replace("$(jq -r '.explainability.magnitude_rate' docs/numbers.json)", str(explainability.get("magnitude_rate", 0.0)))
    
    # Artifacts
    artifacts = numbers.get("artifacts", {})
    template = template.replace("$(jq -r '.artifacts.reports | length' docs/numbers.json)", str(len(artifacts.get("reports", []))))
    template = template.replace("$(jq -r '.artifacts.csv_files | length' docs/numbers.json)", str(len(artifacts.get("csv_files", []))))
    template = template.replace("$(jq -r '.artifacts.parquet_files | length' docs/numbers.json)", str(len(artifacts.get("parquet_files", []))))
    template = template.replace("$(jq -r '.artifacts.total_size_bytes' docs/numbers.json)", str(artifacts.get("total_size_bytes", 0)))
    template = template.replace("$(jq -r '.artifacts.first_date' docs/numbers.json)", str(artifacts.get("first_date", "")))
    template = template.replace("$(jq -r '.artifacts.last_date' docs/numbers.json)", str(artifacts.get("last_date", "")))
    
    # Write the rendered report
    with open("docs/audit_report.md", "w") as f:
        f.write(template)
    
    print("Audit report rendered to docs/audit_report.md")


if __name__ == "__main__":
    render_audit_report()
