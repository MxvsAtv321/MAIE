# MAIE Production Audit Report

**Generated**: 2025-10-22T17:29:45.620111Z  
**Commit**: 2715b3fee398c8666235183402ddca42882ac5ab  
**Python**: Python 3.13.5  
**OS**: Darwin arm64  

## Executive Summary

This audit validates the MAIE quantitative trading system's production readiness through end-to-end testing, performance measurement, and constraint verification.

## Run Metadata

- **Commit SHA**: 2715b3fee398c8666235183402ddca42882ac5ab
- **Timestamp**: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- **Python Version**: Python 3.13.5
- **OS**: Darwin arm64
- **CPU**: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
- **Key Package Versions**: See `docs/numbers.json`

## Expected Panel Facts

- **Shape**: [1, 800]
- **Time Span**: 2024-12-31 00:00:00 to 2024-12-31 00:00:00
- **File Count**: 85
- **Total Bytes**: 49280184
- **Build Time**: 0.0 seconds

## Backtest Metrics

### Unconstrained Backtest
- **Sharpe Ratio (Annualized)**: -3.3844564489065965
- **Volatility (Annualized)**: 0.003384456448905921
- **CAGR**: -0.011389449291846487
- **Max Drawdown**: -0.0009999999999998
- **Turnover (%/day)**: 0.0
- **Avg Gross Exposure**: 0.0
- **Hit Ratio**: 0.0
- **Trades/day**: 0.0

### Constrained Backtest
- **Sharpe Ratio (Annualized)**: -3.3844564489065965
- **Volatility (Annualized)**: 0.003384456448905921
- **CAGR**: -0.011389449291846487
- **Max Drawdown**: -0.0009999999999998
- **Turnover (%/day)**: 0.0
- **Avg Gross Exposure**: 0.0
- **Hit Ratio**: 0.0
- **Trades/day**: 0.0

## Constraint Residuals

- **Max |Net Exposure|**: 0.0046160679634165
- **Mean |Net Exposure|**: 0.00020982127106438636
- **Max |β - Target|**: 0.0032994042793812
- **Mean |β - Target|**: 0.00014997292179005456
- **Max Sector L2**: 0.0023495940440238
- **Mean Sector L2**: 0.00010679972927380908
- **Infeasible Days**: 0 (0.0% of total)

## API Performance

- **/score_expected Median Latency**: 0.0ms
- **/score_expected P95 Latency**: 0.0ms
- **/score_expected Error Rate**: 0.0%
- **/explain_local Median Latency**: 0.0ms
- **/explain_local P95 Latency**: 0.0ms
- **/explain_local Error Rate**: 0.0%

## Explainability Check

- **Non-empty Results**: 100.0%
- **Pred_contrib Success Rate**: 0.0%
- **TreeExplainer Fallback Rate**: 0.0%
- **Magnitude Fallback Rate**: 0.0%

## Artifacts

- **Reports Generated**: 1
- **CSV Files**: 253
- **Parquet Files**: 0
- **Total Size**: 6639932 bytes
- **First Date**: 2024-01-01
- **Last Date**: 2024-12-31

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
