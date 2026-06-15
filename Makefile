.PHONY: install lint typecheck test cov all clean

PY ?= $(shell command -v python3.11 || command -v python3.12 || command -v python3.13 || command -v python3)
VENV ?= .venv
BIN  := $(VENV)/bin

install:
	@$(PY) -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)' \
		|| { echo "Error: $(PY) is $$($(PY) --version 2>&1), but genie needs >=3.11"; exit 1; }
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check genie tests
	$(BIN)/ruff format --check genie tests

format:
	$(BIN)/ruff format genie tests
	$(BIN)/ruff check --fix genie tests

typecheck:
	$(BIN)/pyright

test:
	$(BIN)/pytest

cov:
	$(BIN)/pytest --cov=genie --cov-report=term-missing --cov-fail-under=70

all: lint typecheck cov

clean:
	rm -rf $(VENV) build dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
