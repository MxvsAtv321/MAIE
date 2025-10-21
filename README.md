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
