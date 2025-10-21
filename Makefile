.PHONY: venv install lint test api demo build-expected bt-constrained report report-html

venv:
	uv venv .venv
	. .venv/bin/activate && uv pip install --upgrade pip wheel setuptools

install:
	. .venv/bin/activate && uv pip install -e .[dev]
	pre-commit install

lint:
	ruff check .
	mypy src

test:
	pytest -q

api:
	uvicorn services.api.main:app --reload --port 8000

demo:
	python scripts/run_demo.py

# Build monthly expected returns parquet files + latest snapshot
build-expected:
	python scripts/build_expected_panel.py

# Run constrained backtest using parquet expected panel
bt-constrained: build-expected
	python scripts/run_bt_from_expected.py

# Aggregate monthly outputs into a single CSV report (returns + diagnostics)
report:
	python scripts/make_report.py

# Generate HTML PM report with charts and diagnostics
report-html: report
	python scripts/report_html.py


