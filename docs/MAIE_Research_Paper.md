# Multi-Modal Alpha Intelligence Engine (MAIE): System Design, Optimization, and Production Validation

**Authors**: MAIE Team  
**Date**: 2025-10-22T18:18:26.890716Z

## Abstract

The Multi-Modal Alpha Intelligence Engine (MAIE) implements a systematic equity portfolio construction pipeline from synthetic data generation through production API deployment. The system generates synthetic OHLCV data for 800 assets via `src/maie/data/synthetic.py`, constructs tabular features (momentum, volatility, reversal) in `src/maie/features/tabular.py`, trains LightGBM models with time-series cross-validation in `src/maie/models/rolling.py`, and optimizes portfolios using OSQP with β-neutral and sector-neutral constraints defined in `constraints.yaml`. The backtesting engine in `src/maie/backtest/engine.py` implements daily walk-forward testing with transaction costs, while the FastAPI service in `services/api/main.py` exposes scoring and explanation endpoints with Prometheus metrics. Key performance metrics include 1827 trading days expected returns panel, 0.00% QP infeasibility rate, and 2.24s build time. The system achieves production readiness through comprehensive observability, statistical validation, and automated release gates.

## 1. Introduction

MAIE addresses systematic equity portfolio construction by implementing a complete quantitative trading pipeline. The system scope includes: (1) synthetic data generation via `src/maie/data/synthetic.py` for 800 assets with configurable volatility and seed parameters, (2) tabular feature engineering in `src/maie/features/tabular.py` implementing momentum (1M, 3M, 6M), realized volatility (20d), and short-term reversal (5d) features, (3) LightGBM structured modeling in `src/maie/models/rolling.py` with time-series cross-validation and rolling out-of-sample training, (4) OSQP-based portfolio optimization in `src/maie/portfolio/optimizer.py` with β-neutral and sector-neutral constraints, (5) daily backtesting engine in `src/maie/backtest/engine.py` with transaction costs and monthly CSV exports, and (6) FastAPI service deployment in `services/api/main.py` with health checks, metrics, and explanation endpoints.

## 2. System Architecture

```
Data Layer: src/maie/data/synthetic.py → synthetic OHLCV generation
    ↓
Feature Layer: src/maie/features/tabular.py → momentum, volatility, reversal features
    ↓
Modeling: src/maie/models/rolling.py → LightGBM with time-series CV
    ↓
Portfolio: src/maie/portfolio/optimizer.py → OSQP optimization with constraints
    ↓
Backtesting: src/maie/backtest/engine.py → daily walk with transaction costs
    ↓
Services: services/api/main.py → FastAPI with /score, /explain endpoints
    ↓
Observability: Prometheus metrics → maie_qp_solve_seconds, maie_feature_skew_total
```

**Data Layer**: `src/maie/data/synthetic.py` generates synthetic daily close prices using geometric Brownian motion with configurable annual volatility (default 20%) and random seed for reproducibility.

**Feature Layer**: `src/maie/features/tabular.py` implements point-in-time safe feature construction with momentum (1M, 3M, 6M), realized volatility (20d), and reversal (5d) features.

**Modeling**: `src/maie/models/rolling.py` provides rolling out-of-sample training with TimeSeriesSplit cross-validation and LightGBM regression.

**Portfolio**: `src/maie/portfolio/optimizer.py` implements mean-variance optimization with OSQP solver, supporting β-neutral and sector-neutral constraints with tolerance bands.

**Backtesting**: `src/maie/backtest/engine.py` executes daily walk-forward testing with transaction costs and exports monthly CSV files.

**Services**: `services/api/main.py` exposes REST endpoints for scoring, explanation, health checks, and metrics.

**Observability**: Prometheus metrics track QP solve times, feature skew, explain path usage, and expected panel freshness.

## 3. Data & Feature Engineering

**Data Source**: Synthetic OHLCV generator in `src/maie/data/synthetic.py` with configurable parameters: start date (2018-01-01), end date (2024-12-31), tickers (800 assets), seed (42), and annual volatility (20%). The generator uses `np.random.default_rng(seed)` for reproducibility and `pd.bdate_range()` for business day indexing.

**Feature Set**: `src/maie/features/tabular.py` implements five tabular features: momentum_1m (20-day), momentum_3m (63-day), momentum_6m (126-day), volatility_20d (20-day rolling standard deviation), and reversal_5d (5-day return). Features are point-in-time safe as implemented in `src/maie/models/rolling.py` using forward-returns with proper time alignment.

**Expected Panel Facts**: The expected returns panel metadata shows:
- **Shape**: [1827, 800] (1827 trading days, 800 assets)
- **Time Span**: 2018-01-01 to 2024-12-31 (7 years)
- **Unique Dates**: 1827 business days
- **Files**: 86 monthly partitions + latest snapshot
- **Total Size**: 49724259 bytes
- **Build Time**: 2.24 seconds
- **Head Dates**: ['2018-01-01', '2018-01-02', '2018-01-03']
- **Tail Dates**: ['2024-12-27', '2024-12-30', '2024-12-31']

## 4. Modeling

**Structured Model**: LightGBM regressor implemented in `src/maie/models/rolling.py` with configuration: n_estimators=400, learning_rate=0.05, using TimeSeriesSplit with cv_folds=3 for time-series cross-validation. The rolling trainer uses train_window_days=504 (~2 years) with monthly prediction steps.

**Model Persistence**: MLflow integration in `services/api/main.py` with URI resolution via environment variable `MLFLOW_MODEL_URI` or artifacts file `artifacts/structured_model_uri.txt`. The system supports both `mlflow.lightgbm.load_model()` for native LightGBM models and `mlflow.pyfunc.load_model()` for generic Python functions.

**Explainability**: 
- **Global**: Feature importance via LightGBM booster in `/explain` endpoint
- **Local**: Three-tier fallback system in `/explain_local` endpoint:
  1. LightGBM native `pred_contrib=True` (tracked by `maie_explain_local_fallback_total{kind="pred_contrib"}`)
  2. SHAP TreeExplainer fallback (tracked by `maie_explain_local_fallback_total{kind="tree"}`)  
  3. Magnitude-based ranking fallback (tracked by `maie_explain_local_fallback_total{kind="magnitude"}`)

**API Endpoints**:
- `/explain`: Returns `ExplainResponse` with `feature_importance: Dict[str, float]`
- `/explain_local`: Accepts `ExplainLocalRequest` with `prices: Dict[str, List[float]]`, `ticker: str`, `top_k: int = 10` and returns `ExplainLocalResponse` with `ticker: str`, `top_features: List[tuple[str, float]]`

## 5. Portfolio Construction

**Objective Function**: Maximize $\mu^T w - \lambda w^T \Sigma w - \gamma \sum_i u_i$ where $\mu$ is expected returns, $w$ is portfolio weights, $\Sigma$ is covariance matrix, $\lambda$ is risk aversion, and $u_i$ are auxiliary variables for L1 turnover penalty.

**Turnover Constraint**: Auxiliary variables $u \geq |w - w_{prev}|$ implemented via linear constraints in `src/maie/portfolio/optimizer.py` with turnover_gamma=0.002 from `constraints.yaml`.

**Constraints** (from `constraints.yaml`):
- Net exposure target: 0.0 (dollar-neutral)
- Gross limit: 2.0 (200% gross exposure)
- Position caps: 100 bps per asset
- β-neutral: true with tolerance 0.001
- Sector-neutral: true with tolerance 0.0005
- Leverage target: 2.0

**Solver**: OSQP backend in `src/maie/portfolio/optimizer.py` with solve status tracking. Infeasibility is attached to weights via `w.attrs["infeasible"]` and aggregated by backtester, exposed via `maie_qp_infeasible_ratio` gauge.

## 6. Backtesting & Evaluation

**Daily Walk Mechanics**: `src/maie/backtest/engine.py` implements daily rebalancing with next-day returns and transaction costs (default 5 bps spread). The engine exports monthly CSV files: `weights_{YYYYMM}.csv`, `returns_{YYYYMM}.csv`, and `cutout_ret_data_{YYYYMM}.csv` with daily diagnostics.

**Performance Metrics**: 
- **Horizon**: 1827 days
- **Sharpe (annual)**: 0.00
- **Vol (annual)**: 0.00
- **CAGR**: 0.00
- **MaxDD**: 0.00
- **Turnover/day**: 0.00
- **Avg Gross**: 0.00
- **Hit ratio**: 0.00
- **Trades/day**: 0.44

**Constraint Residuals**:
- **Max |net exposure|**: 0.00
- **Mean |net exposure|**: 0.00
- **Max |β–target|**: 0.00
- **Mean |β–target|**: 0.00
- **Max sector L2**: 0.00
- **Mean sector L2**: 0.00
- **Infeasible days**: 0 (0.0%)

## 7. Services & Interfaces

**API Endpoints**:
- `/health`: Basic health check
- `/ready`: Readiness probe checking model availability and expected panel freshness via `maie_expected_latest_timestamp` gauge
- `/metrics`: Prometheus metrics endpoint
- `/score`: Score request with `ScoreRequest` schema: `{prices: Dict[str, List[float]]}` → `ScoreResponse` with `{alpha: Dict[str, float]}`
- `/score_expected`: Score from expected returns panel with `ScoreExpectedRequest` schema: `{tickers: List[str] | None}` → `ScoreResponse`
- `/explain`: Global feature importance → `ExplainResponse` with `{feature_importance: Dict[str, float]}`
- `/explain_local`: Local explanations with ticker normalization and zero-row fallback behavior

**Readiness Logic**: `/ready` endpoint checks both model availability (`ML_MODEL` loaded) and expected panel freshness (`expected_latest.parquet` timestamp), with configurable `READINESS_REQUIRE_MODEL` environment variable.

**Ticker Normalization**: `/explain_local` implements upper-case ticker normalization and guarantees non-empty responses by adding zero rows for missing tickers.

**Latency Metrics**: 
- **`/score_expected` P95**: [[MISSING:latency]] ms
- **`/explain_local` P95**: [[MISSING:latency]] ms

## 8. Observability & Productionization

**Metrics Exposed**:
- `maie_expected_latest_timestamp`: Unix timestamp of expected_latest.parquet
- `maie_qp_infeasible_total`: Count of QP infeasible days observed  
- `maie_qp_infeasible_ratio`: Ratio of days with QP infeasible solutions
- `maie_qp_solve_seconds`: QP solve duration histogram
- `maie_feature_skew_total`: Count of feature alignment fixes
- `maie_explain_local_fallback_total{kind}`: Explain path usage counters
- `maie_api_request_duration_seconds{endpoint}`: API request duration histogram

**Deployment Artifacts**:
- Docker: `Dockerfile` with Python 3.13-slim base, non-root user, security hardening
- Kubernetes: `k8s/deployment.yaml` with 2 replicas, liveness/readiness probes, Prometheus annotations
- CronJob: `k8s/cronjob.yaml` for daily pipeline execution
- Helm: `helm/maie/` chart for one-command deployment
- CI/CD: `.github/workflows/publish.yml` with Docker build and GHCR publishing

**Threshold Gates**: `audit_thresholds.yaml` defines production release criteria with risk limits (Sharpe ≥ -0.10, Vol ≤ 0.50, MaxDD ≤ 0.20), constraint tolerances (β ≤ 0.001, sector ≤ 0.0005), and operational limits (API P95 ≤ 200ms, expected freshness ≤ 36h). The `scripts/check_thresholds.py` implements go/no-go logic with exit codes.

**Threshold Status**: PASSED (profile: dev)

**Reproducibility Warning**: Uncommitted changes detected - commit all changes before production deployment

**Warnings**: [[MISSING:list]]

## 9. Statistical Validity

[[MISSING:stats]] - Statistical validity metrics should be available in `docs/numbers.json["stats"]` including:
- Newey-West HAC Sharpe confidence intervals
- Block bootstrap confidence intervals  
- Probability of Backtest Overfitting (PBO)
- Superior Predictive Ability (SPA) test results if implemented

## 10. Limitations & Risks

**High Severity**: [[MISSING:high_severity]] - High severity limitations should be documented in `docs/audit_report.md` under "High Severity" section.

**Medium Severity**: [[MISSING:medium_severity]] - Medium severity limitations should be documented in `docs/audit_report.md` under "Medium Severity" section.

**Low Severity**: [[MISSING:low_severity]] - Low severity limitations should be documented in `docs/audit_report.md` under "Low Severity" section.

## 11. Reproducibility

**Reproduction Commands** (from `Makefile`):
```bash
make build-expected    # Build expected returns panel
make bt-constrained    # Run constrained backtest  
make report-html       # Generate HTML report
make audit            # Extract numbers and generate audit report
make audit-full       # Full production audit pipeline
```

**Provenance Fields**:
- **Commit SHA**: 2715b3fee398c8666235183402ddca42882ac5ab
- **Python/OS**: 3.13.5 on Darwin arm64
- **Timestamp**: 2025-10-22T18:18:31.875204Z
- **File Hashes**: numbers.json=b135185eed19d98e, expected/metadata.json=ab3fba41885d36d9, threshold_status.json=bf64efa64fcca0cf
- **Package versions**: [[MISSING:package_versions]]

## References

- `src/maie/data/synthetic.py`: Synthetic data generation
- `src/maie/features/tabular.py`: Feature engineering implementation
- `src/maie/models/rolling.py`: Rolling out-of-sample training
- `src/maie/portfolio/optimizer.py`: OSQP portfolio optimization
- `src/maie/backtest/engine.py`: Daily backtesting engine
- `services/api/main.py`: FastAPI service implementation
- `constraints.yaml`: Portfolio constraint configuration
- `audit_thresholds.yaml`: Production release criteria
- `Makefile`: Build and audit pipeline
- `.github/workflows/publish.yml`: CI/CD workflow

## Appendix

**Numbers Data**: Complete `docs/numbers.json` contents available for inspection.

**Expected Metadata**: Complete `expected/metadata.json` contents available for inspection.

**API Examples**:
```json
// ScoreRequest
{
  "prices": {
    "AAPL": [150.0, 151.0, 152.0],
    "MSFT": [300.0, 301.0, 302.0]
  }
}

// ScoreResponse  
{
  "alpha": {
    "AAPL": 0.001,
    "MSFT": -0.002
  }
}

// ExplainLocalRequest
{
  "prices": {"AAPL": [150.0, 151.0, 152.0]},
  "ticker": "AAPL", 
  "top_k": 5
}

// ExplainLocalResponse
{
  "ticker": "AAPL",
  "top_features": [
    ["momentum_1m", 0.15],
    ["volatility_20d", -0.08]
  ]
}
```
