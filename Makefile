.PHONY: venv install lint test api demo

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


