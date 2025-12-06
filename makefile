# Makefile at repo root

PYTHON := python

.PHONY: help
help:
	@echo "Haven automation targets:"
	@echo "  make dev       - install dev deps, run linters/tests"
	@echo "  make api       - run FastAPI backend (uvicorn)"
	@echo "  make frontend  - run Vite frontend"
	@echo "  make refresh   - refresh data for default ZIPs"
	@echo "  make train     - train all models"
	@echo "  make eval      - evaluate all models"
	@echo "  make full      - full pipeline: refresh + train + eval"
	@echo "  make backtest  - run engine backtest"

.PHONY: dev
dev:
	$(PYTHON) -m pip install -r requirements-dev.txt
	ruff check .
	mypy .
	pytest

.PHONY: api
api:
	uvicorn haven.api.http:app --reload --port 8000

.PHONY: frontend
frontend:
	cd frontend && npm install && npm run dev

# Default zips; adjust as needed or override on the command line:
ZIPS ?= 48009 48363
MAX_PRICE ?= 800000

.PHONY: refresh
refresh:
	$(PYTHON) -m entrypoints.cli.pipeline refresh-data \
		$(foreach z,$(ZIPS),--zip $(z)) \
		--max-price $(MAX_PRICE)

.PHONY: train
train:
	$(PYTHON) -m entrypoints.cli.pipeline train-models

.PHONY: eval
eval:
	$(PYTHON) -m entrypoints.cli.pipeline eval-models

.PHONY: full
full:
	$(PYTHON) -m entrypoints.cli.pipeline full-refresh \
		$(foreach z,$(ZIPS),--zip $(z)) \
		--max-price $(MAX_PRICE)

.PHONY: backtest
backtest:
	$(PYTHON) -m entrypoints.cli.pipeline backtest
