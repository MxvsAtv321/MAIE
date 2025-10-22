# MAIE Production Audit Report

**Generated**: $(date)  
**Commit**: $(git rev-parse HEAD)  
**Python**: $(python --version)  
**OS**: $(uname -s) $(uname -m)  

## Executive Summary

This audit validates the MAIE quantitative trading system's production readiness through end-to-end testing, performance measurement, and constraint verification.

## Run Metadata

- **Commit SHA**: $(git rev-parse HEAD)
- **Timestamp**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- **Python Version**: $(python --version)
- **OS**: $(uname -s) $(uname -m)
- **CPU**: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
- **Key Package Versions**: See `docs/numbers.json`

## Expected Panel Facts

- **Shape**: $(jq -r '.expected_panel.shape' docs/numbers.json)
- **Time Span**: $(jq -r '.expected_panel.start_date' docs/numbers.json) to $(jq -r '.expected_panel.end_date' docs/numbers.json)
- **File Count**: $(jq -r '.expected_panel.file_count' docs/numbers.json)
- **Total Bytes**: $(jq -r '.expected_panel.total_bytes' docs/numbers.json)
- **Build Time**: $(jq -r '.expected_panel.build_time_seconds' docs/numbers.json) seconds

## Backtest Metrics

### Unconstrained Backtest
- **Sharpe Ratio (Annualized)**: $(jq -r '.backtest.unconstrained.sharpe_annual' docs/numbers.json)
- **Volatility (Annualized)**: $(jq -r '.backtest.unconstrained.vol_annual' docs/numbers.json)
- **CAGR**: $(jq -r '.backtest.unconstrained.cagr' docs/numbers.json)
- **Max Drawdown**: $(jq -r '.backtest.unconstrained.max_dd' docs/numbers.json)
- **Turnover (%/day)**: $(jq -r '.backtest.unconstrained.turnover_pct_day' docs/numbers.json)
- **Avg Gross Exposure**: $(jq -r '.backtest.unconstrained.avg_gross' docs/numbers.json)
- **Hit Ratio**: $(jq -r '.backtest.unconstrained.hit_ratio' docs/numbers.json)
- **Trades/day**: $(jq -r '.backtest.unconstrained.trades_per_day' docs/numbers.json)

### Constrained Backtest
- **Sharpe Ratio (Annualized)**: $(jq -r '.backtest.constrained.sharpe_annual' docs/numbers.json)
- **Volatility (Annualized)**: $(jq -r '.backtest.constrained.vol_annual' docs/numbers.json)
- **CAGR**: $(jq -r '.backtest.constrained.cagr' docs/numbers.json)
- **Max Drawdown**: $(jq -r '.backtest.constrained.max_dd' docs/numbers.json)
- **Turnover (%/day)**: $(jq -r '.backtest.constrained.turnover_pct_day' docs/numbers.json)
- **Avg Gross Exposure**: $(jq -r '.backtest.constrained.avg_gross' docs/numbers.json)
- **Hit Ratio**: $(jq -r '.backtest.constrained.hit_ratio' docs/numbers.json)
- **Trades/day**: $(jq -r '.backtest.constrained.trades_per_day' docs/numbers.json)

## Constraint Residuals

- **Max |Net Exposure|**: $(jq -r '.constraints.max_net_exposure' docs/numbers.json)
- **Mean |Net Exposure|**: $(jq -r '.constraints.mean_net_exposure' docs/numbers.json)
- **Max |β - Target|**: $(jq -r '.constraints.max_beta_deviation' docs/numbers.json)
- **Mean |β - Target|**: $(jq -r '.constraints.mean_beta_deviation' docs/numbers.json)
- **Max Sector L2**: $(jq -r '.constraints.max_sector_l2' docs/numbers.json)
- **Mean Sector L2**: $(jq -r '.constraints.mean_sector_l2' docs/numbers.json)
- **Infeasible Days**: $(jq -r '.constraints.infeasible_days' docs/numbers.json) ($(jq -r '.constraints.infeasible_pct' docs/numbers.json)% of total)

## API Performance

- **/score_expected Median Latency**: $(jq -r '.api.score_expected.median_ms' docs/numbers.json)ms
- **/score_expected P95 Latency**: $(jq -r '.api.score_expected.p95_ms' docs/numbers.json)ms
- **/score_expected Error Rate**: $(jq -r '.api.score_expected.error_rate' docs/numbers.json)%
- **/explain_local Median Latency**: $(jq -r '.api.explain_local.median_ms' docs/numbers.json)ms
- **/explain_local P95 Latency**: $(jq -r '.api.explain_local.p95_ms' docs/numbers.json)ms
- **/explain_local Error Rate**: $(jq -r '.api.explain_local.error_rate' docs/numbers.json)%

## Explainability Check

- **Non-empty Results**: $(jq -r '.explainability.non_empty_rate' docs/numbers.json)%
- **Pred_contrib Success Rate**: $(jq -r '.explainability.pred_contrib_rate' docs/numbers.json)%
- **TreeExplainer Fallback Rate**: $(jq -r '.explainability.tree_explainer_rate' docs/numbers.json)%
- **Magnitude Fallback Rate**: $(jq -r '.explainability.magnitude_rate' docs/numbers.json)%

## Artifacts

- **Reports Generated**: $(jq -r '.artifacts.reports | length' docs/numbers.json)
- **CSV Files**: $(jq -r '.artifacts.csv_files | length' docs/numbers.json)
- **Parquet Files**: $(jq -r '.artifacts.parquet_files | length' docs/numbers.json)
- **Total Size**: $(jq -r '.artifacts.total_size_bytes' docs/numbers.json) bytes
- **First Date**: $(jq -r '.artifacts.first_date' docs/numbers.json)
- **Last Date**: $(jq -r '.artifacts.last_date' docs/numbers.json)

## Known Limitations & Risks

### High Severity
- **None identified**
{{ "**Dirty Git Tree**: Uncommitted changes detected - commit all changes before production deployment" if dirty_tree else "" }}

### Medium Severity
- **None identified**

### Low Severity
- **None identified**

## Next Actions

- **Owner**: [Team Lead]
- **ETA**: [Date]
- **Priority**: [High/Medium/Low]

---

*This report was generated automatically by the MAIE audit system.*
