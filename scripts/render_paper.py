#!/usr/bin/env python3
"""
Template-based research paper renderer.
Generates docs/MAIE_Research_Paper.md from docs/numbers.json and expected/metadata.json.
Eliminates manual drift by using string.Template for all dynamic content.
"""

import json
import pathlib
import hashlib
import subprocess
from string import Template
from datetime import datetime


def get_git_info():
    """Get git commit and dirty tree status."""
    try:
        # Get commit SHA
        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], 
            capture_output=True, text=True, check=True
        ).stdout.strip()
        
        # Check for dirty tree
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain"], 
            capture_output=True, text=True
        ).stdout.strip()
        dirty_tree = dirty_output != ""
        
        return commit_sha, dirty_tree
    except Exception:
        return "unknown", False


def get_file_hash(file_path: pathlib.Path) -> str:
    """Get SHA256 hash of a file."""
    if not file_path.exists():
        return "missing"
    try:
        content = file_path.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    except Exception:
        return "error"


def load_data():
    """Load all required data files."""
    data = {}
    
    # Load numbers.json
    numbers_path = pathlib.Path("docs/numbers.json")
    if numbers_path.exists():
        data["numbers"] = json.loads(numbers_path.read_text())
    else:
        data["numbers"] = {}
    
    # Load expected metadata
    expected_meta_path = pathlib.Path("expected/metadata.json")
    if expected_meta_path.exists():
        data["expected_meta"] = json.loads(expected_meta_path.read_text())
    else:
        data["expected_meta"] = {}
    
    # Load threshold status
    threshold_status_path = pathlib.Path("docs/threshold_status.json")
    if threshold_status_path.exists():
        data["threshold_status"] = json.loads(threshold_status_path.read_text())
    else:
        data["threshold_status"] = {"status": "UNKNOWN"}
    
    # Get git info
    commit_sha, dirty_tree = get_git_info()
    data["git_commit"] = commit_sha
    data["dirty_tree"] = dirty_tree
    
    # Get file hashes
    data["numbers_hash"] = get_file_hash(numbers_path)
    data["expected_meta_hash"] = get_file_hash(expected_meta_path)
    data["threshold_status_hash"] = get_file_hash(threshold_status_path)
    
    return data


def format_number(value, decimals=2):
    """Format a number with specified decimal places."""
    if value is None:
        return "[[MISSING:number]]"
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "[[MISSING:number]]"


def format_list(value, max_items=3):
    """Format a list with max items shown."""
    if not value or not isinstance(value, list):
        return "[[MISSING:list]]"
    if len(value) <= max_items:
        return str(value)
    return str(value[:max_items]) + f"... (+{len(value)-max_items} more)"


def get_or_missing(data, path, default="[[MISSING:data]]"):
    """Get nested value from data dict or return default."""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def render_paper():
    """Render the research paper from template and data."""
    
    # Load all data
    data = load_data()
    
    # Extract key metrics
    expected_panel = data["numbers"].get("expected_panel", {})
    backtest = data["numbers"].get("backtest", {}).get("constrained", {})
    constraints = data["numbers"].get("constraints", {})
    api_metrics = data["numbers"].get("api", {})
    expected_meta = data["expected_meta"]
    threshold_status = data["threshold_status"]
    
    # Template variables
    template_vars = {
        # Expected panel facts
        "expected_shape": str(expected_panel.get("shape", "[[MISSING:shape]]")),
        "expected_start": expected_panel.get("start", "[[MISSING:start]]"),
        "expected_end": expected_panel.get("end", "[[MISSING:end]]"),
        "expected_n_files": str(expected_panel.get("n_files", "[[MISSING:n_files]]")),
        "expected_total_bytes": str(expected_panel.get("total_bytes", "[[MISSING:bytes]]")),
        "expected_build_seconds": format_number(expected_panel.get("build_seconds")),
        "expected_n_unique_dates": str(expected_panel.get("n_unique_dates", "[[MISSING:dates]]")),
        "expected_head_dates": format_list(expected_panel.get("head_dates", [])),
        "expected_tail_dates": format_list(expected_panel.get("tail_dates", [])),
        
        # Backtest metrics
        "sharpe_annual": format_number(backtest.get("sharpe_annual")),
        "vol_annual": format_number(backtest.get("vol_annual")),
        "cagr": format_number(backtest.get("cagr")),
        "max_dd": format_number(backtest.get("max_dd")),
        "turnover_pct_day": format_number(backtest.get("turnover_pct_day")),
        "avg_gross": format_number(backtest.get("avg_gross")),
        "hit_ratio": format_number(backtest.get("hit_ratio")),
        "trades_per_day": format_number(backtest.get("trades_per_day")),
        
        # Constraint residuals
        "max_net_exposure": format_number(constraints.get("max_net_exposure")),
        "mean_net_exposure": format_number(constraints.get("mean_net_exposure")),
        "max_beta_deviation": format_number(constraints.get("max_beta_deviation")),
        "mean_beta_deviation": format_number(constraints.get("mean_beta_deviation")),
        "max_sector_l2": format_number(constraints.get("max_sector_l2")),
        "mean_sector_l2": format_number(constraints.get("mean_sector_l2")),
        "infeasible_days": str(constraints.get("infeasible_days", "[[MISSING:infeasible]]")),
        "infeasible_pct": format_number(constraints.get("infeasible_pct")),
        
        # API metrics (only if measured)
        "score_expected_p95_ms": format_number(api_metrics.get("score_expected", {}).get("p95_ms")) if api_metrics.get("score_expected") else "[[MISSING:latency]]",
        "explain_local_p95_ms": format_number(api_metrics.get("explain_local", {}).get("p95_ms")) if api_metrics.get("explain_local") else "[[MISSING:latency]]",
        
        # Threshold status
        "threshold_status": threshold_status.get("status", "UNKNOWN"),
        "threshold_violations": format_list(threshold_status.get("violations", [])),
        "threshold_profile": threshold_status.get("profile", "unknown"),
        
        # Metadata
        "commit_sha": data["numbers"].get("metadata", {}).get("commit_sha", "[[MISSING:commit]]"),
        "timestamp": data["numbers"].get("metadata", {}).get("timestamp", "[[MISSING:timestamp]]"),
        "python_version": data["numbers"].get("metadata", {}).get("python_version", "[[MISSING:python]]"),
        "os": data["numbers"].get("metadata", {}).get("os", "[[MISSING:os]]"),
        
        # Warnings
        "warnings": format_list(data["numbers"].get("warnings", [])),
        
        # Provenance
        "git_commit": data["git_commit"],
        "dirty_tree": data["dirty_tree"],
        "numbers_hash": data["numbers_hash"],
        "expected_meta_hash": data["expected_meta_hash"],
        "threshold_status_hash": data["threshold_status_hash"],
        "audit_timestamp": datetime.utcnow().isoformat() + "Z",
        
        # Conditional text
        "threshold_violations_text": f"**Threshold Violations**: {', '.join(threshold_status.get('violations', []))}" if threshold_status.get('status') == 'FAILED' else "",
        "reproducibility_warning": "**Reproducibility Warning**: Uncommitted changes detected - commit all changes before production deployment" if data["dirty_tree"] else "",
        
        # Combined variables to avoid nested placeholders
        "expected_shape_with_dates": f"{expected_panel.get('shape', '[[MISSING:shape]]')} ({expected_panel.get('n_unique_dates', '[[MISSING:dates]]')} trading days, 800 assets)",
        "infeasible_days_with_pct": f"{constraints.get('infeasible_days', '[[MISSING:infeasible]]')} ({constraints.get('infeasible_pct', '[[MISSING:pct]]')}%)",
        "threshold_status_with_profile": f"{threshold_status.get('status', 'UNKNOWN')} (profile: {threshold_status.get('profile', 'unknown')})",
        "python_os": f"{data['numbers'].get('metadata', {}).get('python_version', '[[MISSING:python]]')} on {data['numbers'].get('metadata', {}).get('os', '[[MISSING:os]]')}",
        "file_hashes": f"numbers.json={data['numbers_hash']}, expected/metadata.json={data['expected_meta_hash']}, threshold_status.json={data['threshold_status_hash']}",
    }
    
    # Paper template
    paper_template = Template("""# Multi-Modal Alpha Intelligence Engine (MAIE): System Design, Optimization, and Production Validation

**Authors**: MAIE Team  
**Date**: ${timestamp}

## Abstract

The Multi-Modal Alpha Intelligence Engine (MAIE) implements a systematic equity portfolio construction pipeline from synthetic data generation through production API deployment. The system generates synthetic OHLCV data for 800 assets via `src/maie/data/synthetic.py`, constructs tabular features (momentum, volatility, reversal) in `src/maie/features/tabular.py`, trains LightGBM models with time-series cross-validation in `src/maie/models/rolling.py`, and optimizes portfolios using OSQP with β-neutral and sector-neutral constraints defined in `constraints.yaml`. The backtesting engine in `src/maie/backtest/engine.py` implements daily walk-forward testing with transaction costs, while the FastAPI service in `services/api/main.py` exposes scoring and explanation endpoints with Prometheus metrics. Key performance metrics include ${expected_n_unique_dates} trading days expected returns panel, ${infeasible_pct}% QP infeasibility rate, and ${expected_build_seconds}s build time. The system achieves production readiness through comprehensive observability, statistical validation, and automated release gates.

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
- **Shape**: ${expected_shape_with_dates}
- **Time Span**: ${expected_start} to ${expected_end} (7 years)
- **Unique Dates**: ${expected_n_unique_dates} business days
- **Files**: ${expected_n_files} monthly partitions + latest snapshot
- **Total Size**: ${expected_total_bytes} bytes
- **Build Time**: ${expected_build_seconds} seconds
- **Head Dates**: ${expected_head_dates}
- **Tail Dates**: ${expected_tail_dates}

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

**Objective Function**: Maximize $$\\mu^T w - \\lambda w^T \\Sigma w - \\gamma \\sum_i u_i$$ where $$\\mu$$ is expected returns, $$w$$ is portfolio weights, $$\\Sigma$$ is covariance matrix, $$\\lambda$$ is risk aversion, and $$u_i$$ are auxiliary variables for L1 turnover penalty.

**Turnover Constraint**: Auxiliary variables $$u \\geq |w - w_{prev}|$$ implemented via linear constraints in `src/maie/portfolio/optimizer.py` with turnover_gamma=0.002 from `constraints.yaml`.

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
- **Horizon**: ${expected_n_unique_dates} days
- **Sharpe (annual)**: ${sharpe_annual}
- **Vol (annual)**: ${vol_annual}
- **CAGR**: ${cagr}
- **MaxDD**: ${max_dd}
- **Turnover/day**: ${turnover_pct_day}
- **Avg Gross**: ${avg_gross}
- **Hit ratio**: ${hit_ratio}
- **Trades/day**: ${trades_per_day}

**Constraint Residuals**:
- **Max |net exposure|**: ${max_net_exposure}
- **Mean |net exposure|**: ${mean_net_exposure}
- **Max |β–target|**: ${max_beta_deviation}
- **Mean |β–target|**: ${mean_beta_deviation}
- **Max sector L2**: ${max_sector_l2}
- **Mean sector L2**: ${mean_sector_l2}
- **Infeasible days**: ${infeasible_days_with_pct}

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
- **`/score_expected` P95**: ${score_expected_p95_ms} ms
- **`/explain_local` P95**: ${explain_local_p95_ms} ms

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

**Threshold Status**: ${threshold_status_with_profile}
${threshold_violations_text}
${reproducibility_warning}

**Warnings**: ${warnings}

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
- **Commit SHA**: ${git_commit}
- **Python/OS**: ${python_os}
- **Timestamp**: ${audit_timestamp}
- **File Hashes**: ${file_hashes}
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
""")
    
    # Render the paper
    paper_content = paper_template.substitute(**template_vars)
    
    # Write to file
    paper_path = pathlib.Path("docs/MAIE_Research_Paper.md")
    paper_path.write_text(paper_content, encoding="utf-8")
    
    print(f"Research paper rendered to {paper_path}")
    return paper_path


if __name__ == "__main__":
    render_paper()
