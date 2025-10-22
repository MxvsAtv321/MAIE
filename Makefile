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

# Run production audit and extract numbers
audit: bt-constrained
	python scripts/extract_numbers.py
	python scripts/render_audit_report.py
	@echo "Audit completed. Check docs/audit_report.md and docs/numbers.json"

# Run performance tests
perf:
	pytest -q -k "perf" tests/test_perf_local.py

# Run threshold checks
check-thresholds:
	python scripts/check_thresholds.py audit_thresholds.yaml docs/numbers.json

# Run statistical validity checks
stats-validity:
	python scripts/stats_validity.py

# Full production audit pipeline
audit-full: bt-constrained report-html audit stats-validity check-thresholds

# Generate research paper
paper: audit
	python scripts/render_paper.py
	@echo "Paper generated at docs/MAIE_Research_Paper.md"

# Generate release notes
release-notes: audit
	python scripts/generate_release_notes.py
	@echo "Release notes generated at docs/RELEASE_NOTES.md"


