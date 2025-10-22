# MAIE Release Notes

## Version & Provenance

- **Commit SHA**: 2715b3f
- **Profile**: dev
- **Audit Timestamp**: 2025-10-22T18:21:18.413805Z
- **Dirty Tree**: Yes
- **Numbers Hash**: 48643c013196d255

## Summary

- **Expected Panel**: Shape [1827, 800], 2018-01-01 to 2024-12-31, 86 files, 2.27s build time
- **Backtest Horizon**: 1827 trading days
- **QP Infeasible Ratio**: 0.00%
- **Performance Metrics**: Sharpe 0.00, Vol 0.00, MaxDD 0.00
- **API Observability**: Prometheus metrics with custom gauges and counters

## Threshold Gate Verdict

- **Status**: PASSED
- **Violations**: N/A

## Notable Changes

- **Initial Release**: Complete MAIE system implementation
- **CI/CD**: GitHub Actions with Docker publishing to GHCR
- **Metrics**: Prometheus instrumentation with custom gauges and counters
- **Constraints**: β-neutral and sector-neutral with tolerance bands
- **Explainability**: Three-tier fallback system for local explanations

## Risks & Limitations

- **Warnings**: N/A

## Repro Steps

```bash
make build-expected
make bt-constrained
make audit
make paper
```

## Artifacts

- `docs/MAIE_Research_Paper.md`
- `docs/numbers.json`
- `docs/threshold_status.json`
- `expected/metadata.json`
- `outputs_from_expected/report.html`
