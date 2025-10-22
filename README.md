# MAIE

End-to-end MVP: synthetic data  features  model  optimizer  backtester  API.

## Quickstart (Apple Silicon)

```bash
# Tooling
xcode-select --install || true
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || true
brew update
brew install git gh uv python@3.13 cmake libomp openblas pkg-config node@22 pnpm

# Repo
uv venv .venv && source .venv/bin/activate
uv pip install --upgrade pip wheel setuptools
uv pip install -e .
uv pip install pre-commit ruff mypy black pytest
pre-commit install

# Smoke
pytest -q

# Demo
python scripts/run_demo.py

# API
uvicorn services.api.main:app --reload --port 8000
```

## Numbers & SLOs

For production numbers validation and audit reports, see:
- **Audit Report**: `docs/audit_report.md` - Comprehensive production audit with metrics
- **Numbers Data**: `docs/numbers.json` - Machine-readable metrics and KPIs
- **Run Audit**: `make audit` - Extract numbers and generate audit report
- **Performance Tests**: `make perf` - Run performance benchmarks

### Key SLOs
- **API Latency**: `/score_expected` P95 < 200ms, `/explain_local` P95 < 400ms
- **Constraint Satisfaction**: Neutrality within tolerances on ≥99% of days
- **QP Infeasibility**: ≤ 0.1% of backtest days
- **Expected Panel Freshness**: < 36h old
- **Explainability**: Non-empty results for all ticker requests

Open `http://127.0.0.1:8000/docs` and call `POST /score`.

## Structure

```
src/maie/
  data/
  features/
  models/
  portfolio/
  backtest/
services/api/
scripts/
tests/
```

## License
MIT
