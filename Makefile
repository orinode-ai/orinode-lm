SHELL := /bin/bash
.DEFAULT_GOAL := help

# Load .env if it exists (ignored silently if absent)
-include .env
export

PYTHON := uv run python
PYTEST  := uv run pytest
RUFF    := uv run ruff
MYPY    := uv run mypy

# ── help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── setup ─────────────────────────────────────────────────────────────────────
.PHONY: install
install: init-workspace ## Install Python deps (uv sync) + npm ci in ui/frontend/
	uv sync --all-extras
	cd ui/frontend && npm ci

.PHONY: init-workspace
init-workspace: ## Create workspace/ directory tree (idempotent)
	@mkdir -p \
	  workspace/data/raw/naijavoices \
	  workspace/data/raw/afrispeech_200 \
	  workspace/data/raw/bibletts \
	  workspace/data/raw/common_voice \
	  workspace/data/raw/crowdsourced_cs \
	  workspace/data/processed \
	  workspace/data/manifests \
	  workspace/data/filter_cache \
	  workspace/models/base \
	  workspace/models/checkpoints \
	  workspace/logs/wandb \
	  workspace/logs/training \
	  workspace/logs/filter_reports \
	  workspace/evals \
	  workspace/cache/huggingface \
	  workspace/cache/transformers
	@echo "workspace/ initialised"

# ── data ──────────────────────────────────────────────────────────────────────
.PHONY: download-data
download-data: ## Download all corpora to workspace/data/raw/
	$(PYTHON) scripts/data/download_naijavoices.py
	$(PYTHON) scripts/data/download_afrispeech.py
	$(PYTHON) scripts/data/download_bibletts.py
	$(PYTHON) scripts/data/download_common_voice.py

DATASETS_DIR ?= $(or $(ORINODE_DATASETS_DIR),$(HOME)/datasets)
AFRISPEECH_DIR ?= $(DATASETS_DIR)/afrispeech-200/nigeria

.PHONY: download-afrispeech-nigeria
download-afrispeech-nigeria: ## Download Nigerian-only AfriSpeech-200 train split
	$(PYTHON) scripts/deploy/download_afrispeech_nigeria.py --split train

.PHONY: download-afrispeech-nigeria-test
download-afrispeech-nigeria-test: ## Download Nigerian-only AfriSpeech-200 test split
	$(PYTHON) scripts/deploy/download_afrispeech_nigeria.py --split test

.PHONY: download-afrispeech-nigeria-dev
download-afrispeech-nigeria-dev: ## Download Nigerian-only AfriSpeech-200 validation (dev) split
	$(PYTHON) scripts/deploy/download_afrispeech_nigeria.py --split validation

.PHONY: download-afrispeech-nigeria-dry-run
download-afrispeech-nigeria-dry-run: ## Dry-run AfriSpeech train download (preview 5 rows)
	$(PYTHON) scripts/deploy/download_afrispeech_nigeria.py --split train --dry-run

.PHONY: link-datasets
link-datasets: ## Symlink ${ORINODE_DATASETS_DIR:-$HOME/datasets}/* into workspace/data/raw/
	@bash scripts/deploy/link_datasets.sh

.PHONY: filter-afrispeech-metadata
filter-afrispeech-metadata: ## Pass 1 — metadata filter on train split (~10 min)
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_train.json \
	  --split-name train --filter-level metadata

.PHONY: filter-afrispeech-metadata-test
filter-afrispeech-metadata-test: ## Pass 1 — metadata filter on test split
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_test.json \
	  --split-name test --filter-level metadata

.PHONY: filter-afrispeech-metadata-dev
filter-afrispeech-metadata-dev: ## Pass 1 — metadata filter on validation (dev) split
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_validation.json \
	  --split-name validation --filter-level metadata

.PHONY: filter-afrispeech-audio
filter-afrispeech-audio: ## Pass 1+2 — metadata + audio quality on train (~2-3 h)
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_train.json \
	  --split-name train --filter-level audio

.PHONY: filter-afrispeech-audio-dev
filter-afrispeech-audio-dev: ## Pass 1+2 — metadata + audio quality on dev split
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_validation.json \
	  --split-name validation --filter-level audio

.PHONY: filter-afrispeech-audio-test
filter-afrispeech-audio-test: ## Pass 1+2 — metadata + audio quality on test split
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_test.json \
	  --split-name test --filter-level audio

.PHONY: filter-afrispeech-audio-all
filter-afrispeech-audio-all: filter-afrispeech-audio filter-afrispeech-audio-dev filter-afrispeech-audio-test ## Pass 1+2 on all splits sequentially

.PHONY: filter-afrispeech-full
filter-afrispeech-full: ## All 3 passes including Whisper CER on train (~40-50 min with batching)
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_train.json \
	  --split-name train --filter-level full

.PHONY: filter-afrispeech-full-dev
filter-afrispeech-full-dev: ## All 3 passes including Whisper CER on dev split (~5 min)
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_validation.json \
	  --split-name validation --filter-level full

.PHONY: filter-afrispeech-full-test
filter-afrispeech-full-test: ## All 3 passes including Whisper CER on test split (~15 min)
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_test.json \
	  --split-name test --filter-level full

.PHONY: filter-afrispeech-full-all
filter-afrispeech-full-all: filter-afrispeech-full filter-afrispeech-full-dev filter-afrispeech-full-test ## All 3 passes on all splits sequentially (~60 min with batching)

.PHONY: build-manifests
build-manifests: ## Run full pipeline (all passes, train split)
	$(PYTHON) scripts/data/build_manifests.py \
	  --metadata-path $(AFRISPEECH_DIR)/metadata_train.json \
	  --split-name train --filter-level full

.PHONY: validate-data
validate-data: ## Run diacritic validation over all manifests
	$(PYTHON) scripts/data/validate_diacritics.py

# ── aux models ────────────────────────────────────────────────────────────────
.PHONY: train-gender
train-gender: ## Train auxiliary gender classifier
	$(PYTHON) -m orinode.training.train_gender

.PHONY: train-emotion
train-emotion: ## Train auxiliary emotion classifier
	$(PYTHON) -m orinode.training.train_emotion

# ── deploy ────────────────────────────────────────────────────────────────────
.PHONY: vps-bootstrap
vps-bootstrap: ## Bootstrap VPS: deps, CUDA check, firewall
	@bash scripts/deploy/vps_bootstrap.sh

.PHONY: ui-credentials
ui-credentials: ## Generate HTTP basic auth credentials for the UI
	@bash scripts/deploy/generate_ui_credentials.sh

# ── training ──────────────────────────────────────────────────────────────────
.PHONY: smoke-test
smoke-test: ## One forward+backward on synthetic data (CPU OK, ~60 s)
	$(PYTHON) -m orinode.training.stage1_encoder --smoke-test
	@echo "smoke-test PASSED"

.PHONY: smoke-test-real
smoke-test-real: ## Real Whisper large-v3, 8 train clips, 2 steps — validates VRAM budget (GPU required)
	$(PYTHON) -m orinode.training.stage1_encoder --smoke-test-real
	@echo "smoke-test-real PASSED"

.PHONY: smoke-test-real-full
smoke-test-real-full: ## batch=16, DoRA r=32, grad-ckpt, 4 optimizer steps — VRAM budget under real conditions
	$(PYTHON) -m orinode.training.stage1_encoder --smoke-test-real-full
	@echo "smoke-test-real-full PASSED"

.PHONY: train-stage1
train-stage1: ## Stage 1: Whisper encoder DoRA on AfriSpeech Nigerian-English
	bash scripts/train/launch_stage1.sh

.PHONY: train-stage2
train-stage2: ## Stage 2: joint multilingual ASR (all 5 langs + CS)
	bash scripts/train/launch_stage2.sh

.PHONY: train-stage3
train-stage3: ## Stage 3: freeze encoder, train adapter + LoRA decoder
	bash scripts/train/launch_stage3.sh

.PHONY: train-stage4
train-stage4: ## Stage 4: instruction fine-tune (call-center, translation, Q&A)
	bash scripts/train/launch_stage4.sh

# ── eval ──────────────────────────────────────────────────────────────────────
.PHONY: eval
eval: ## Run ASR + CS eval for a run: make eval RUN=<run_id>
	@test -n "$(RUN)" || (echo "Usage: make eval RUN=<run_id>" && exit 1)
	bash scripts/eval/run_asr_eval.sh $(RUN)
	bash scripts/eval/run_cs_eval.sh $(RUN)

.PHONY: leaderboard
leaderboard: ## Build leaderboard across all eval reports
	$(PYTHON) scripts/eval/build_leaderboard.py

# ── UI ────────────────────────────────────────────────────────────────────────
.PHONY: ui-dev
ui-dev: ## Run Vite dev (5173) + uvicorn (7860) with hot reload
	bash scripts/ui/dev.sh

.PHONY: ui-build
ui-build: ## Build React frontend → src/orinode/ui/static/
	bash scripts/ui/build_frontend.sh

.PHONY: ui
ui: ui-build ## Build frontend then serve everything on :7860
	$(PYTHON) -m orinode.ui.server

# ── code quality ──────────────────────────────────────────────────────────────
.PHONY: lint
lint: ## Ruff check + format check
	$(RUFF) check src tests scripts
	$(RUFF) format --check src tests scripts

.PHONY: fmt
fmt: ## Auto-fix lint issues + reformat
	$(RUFF) check --fix src tests scripts
	$(RUFF) format src tests scripts

.PHONY: typecheck
typecheck: ## mypy strict type check
	$(MYPY) src/

.PHONY: test
test: ## Run fast tests (excludes @pytest.mark.slow)
	$(PYTEST) tests/ -m "not slow" -v --tb=short

.PHONY: test-all
test-all: ## Run all tests including slow (needs model weights)
	$(PYTEST) tests/ -v --tb=short

.PHONY: check
check: lint typecheck test ## lint + typecheck + fast tests
