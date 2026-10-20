# Skeleton action recognition on KARD + Xbox 360 Kinect v1.
# Run inside WSL2 Ubuntu (or any Linux) from the project root:  make help
#
# Variables you may override:  make train-stgcn SEED=43 EPOCHS=200
#   VENV          python virtualenv                      (default: ~/venv)
#   KARD_TAR_DIR  folder with the official KARD_*.tar parts
#   CKPT          checkpoint used by infer/realtime targets
#   SEED, EPOCHS  for the single-run training targets

SHELL := /bin/bash
.SHELLFLAGS := -eo pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

VENV          ?= $(HOME)/venv
PY            := $(VENV)/bin/python
PIP           := $(VENV)/bin/pip
KARD_TAR_DIR  ?= /mnt/d/Datasets/KARD - Kinect Activity Recognition Dataset
SEED          ?= 42
EPOCHS        ?=
DEVICE        ?= auto
LOGDIR        := outputs/logs
BRIDGE_PORT   ?= 8765
SMOKE_CKPT    := outputs/checkpoints/smoke_stgcn/seed0/best.pt
MAIN_CKPT     := outputs/checkpoints/e3_stgcn_xyz/seed$(SEED)/best.pt
CKPT          ?= $(shell if [ -f "$(MAIN_CKPT)" ]; then echo "$(MAIN_CKPT)"; else echo "$(SMOKE_CKPT)"; fi)
EPOCH_ARG      = $(if $(EPOCHS),training.epochs=$(EPOCHS),)
WIN_ROOT       = $(shell wslpath -w "$(CURDIR)" 2>/dev/null)

.PHONY: help install check lint format test test-fast extract-data inspect-data prepare-data \
        data smoke train-baseline train-lstm train-stgcn train-all experiments experiments-full loso aggregate \
        evaluate benchmark infer-demo realtime-replay realtime-kinect kinect-test kinect-diag kinect-bridge live \
        notebooks docker-build clean clean-cache clean-outputs env-info

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------------------- setup
install:  ## install/upgrade dependencies into $(VENV) (+ editable package)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e . --no-deps
	$(PY) -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

env-info:  ## print Python / torch / GPU information
	$(PY) -c "import sys,torch,numpy; print(sys.version); print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'); print('numpy', numpy.__version__)"

# ----------------------------------------------------------------------------- quality
lint:  ## ruff lint
	$(VENV)/bin/ruff check src scripts tests

format:  ## ruff format (rewrites files)
	$(VENV)/bin/ruff format src scripts tests
	$(VENV)/bin/ruff check --fix src scripts tests

test:  ## run the test suite
	$(PY) -m pytest

check: lint  ## lint + format check + tests
	$(VENV)/bin/ruff format --check src scripts tests
	$(PY) -m pytest

# ----------------------------------------------------------------------------- data
extract-data:  ## extract *_realworld/_screen.txt from the KARD tar parts into data/raw/kard
	$(PY) scripts/extract_kard.py --tar-dir "$(KARD_TAR_DIR)"

inspect-data:  ## dataset inspection report -> outputs/reports/
	$(PY) scripts/inspect_kard.py --kind realworld

prepare-data:  ## parse + cache interim data and preprocessed feature tensors
	$(PY) scripts/prepare_dataset.py

data: extract-data inspect-data prepare-data  ## extract + inspect + prepare

# ----------------------------------------------------------------------------- training
smoke:  ## 5-epoch pipeline check for all three models (not for reporting)
	mkdir -p $(LOGDIR)
	for m in mlp lstm stgcn; do
	  $(PY) scripts/train.py --config configs/experiments/e3_stgcn_xyz.yaml --model configs/model/$$m.yaml \
	    --training configs/training/smoke.yaml --name smoke_$$m --overwrite --set training.device=$(DEVICE) \
	    2>&1 | tee $(LOGDIR)/smoke_$$m.log || exit 1
	done

define TRAIN_ONE
	mkdir -p $(LOGDIR)
	$(PY) scripts/train.py --config configs/experiments/$(1).yaml \
	  --set training.seed=$(SEED) training.device=$(DEVICE) $(EPOCH_ARG) $(TRAIN_ARGS) \
	  2>&1 | tee $(LOGDIR)/$(1)_seed$(SEED).log
endef

train-baseline:  ## Model A: MLP baseline (e1), full training
	$(call TRAIN_ONE,e1_mlp_xyz)

train-lstm:  ## Model B: BiLSTM (e2), full training
	$(call TRAIN_ONE,e2_lstm_xyz)

train-stgcn:  ## Model C: ST-GCN (e3), full training
	$(call TRAIN_ONE,e3_stgcn_xyz)

train-all: train-baseline train-lstm train-stgcn  ## A + B + C with SEED

experiments:  ## controlled suite, 14 runs (~15 min): A/B/C x 3 seeds + ablations x 1 seed; resumable
	mkdir -p $(LOGDIR)
	$(PY) scripts/run_experiments.py --suite configs/experiments/suite.yaml 2>&1 | tee -a $(LOGDIR)/suite.log

experiments-full:  ## full suite, 24 runs: every experiment x 3 seeds (reuses finished runs)
	mkdir -p $(LOGDIR)
	$(PY) scripts/run_experiments.py --suite configs/experiments/suite_full.yaml 2>&1 | tee -a $(LOGDIR)/suite.log

loso:  ## leave-one-subject-out CV of ST-GCN (10 folds), resumable
	mkdir -p $(LOGDIR)
	$(PY) scripts/run_experiments.py --loso configs/experiments/e3_stgcn_xyz.yaml --seeds $(SEED) \
	  2>&1 | tee -a $(LOGDIR)/loso.log

aggregate:  ## re-aggregate finished runs into outputs/metrics/suite_summary.{csv,md}
	$(PY) scripts/run_experiments.py --aggregate-only

# ----------------------------------------------------------------------------- evaluation
evaluate:  ## evaluate every best.pt on its test subjects (metrics + confusion matrices)
	$(PY) scripts/evaluate.py --all --partition test --device $(DEVICE)

benchmark:  ## params / latency of untrained default models (+ CKPT if it exists)
	if [ -f "$(CKPT)" ]; then $(PY) scripts/benchmark.py --fresh --checkpoint "$(CKPT)"; \
	else $(PY) scripts/benchmark.py --fresh; fi

# ----------------------------------------------------------------------------- inference
infer-demo:  ## offline inference on a raw test-subject file, a cached sequence and a synthetic one
	@test -f "$(CKPT)" || { echo "No checkpoint at $(CKPT). Run 'make smoke' or 'make train-stgcn'."; exit 1; }
	echo "== checkpoint: $(CKPT)"
	$(PY) scripts/infer.py --checkpoint "$(CKPT)" --raw-file data/raw/kard/KARD_a17_a18/a17_s09_e01_realworld.txt --topk 3
	$(PY) scripts/infer.py --checkpoint "$(CKPT)" --seq-id a14_s10_e02 --topk 3
	$(PY) scripts/infer.py --checkpoint "$(CKPT)" --synthetic --topk 3

realtime-replay:  ## mock live stream: replay test-subject KARD sequences in real time (window if display)
	$(PY) scripts/run_realtime.py --checkpoint "$(CKPT)" --source replay --subjects 9 10 --limit 36 $(RT_ARGS)

realtime-kinect:  ## WSL listener for the Windows Kinect bridge (start `make kinect-bridge` separately)
	$(PY) scripts/run_realtime.py --checkpoint "$(CKPT)" --source bridge --port $(BRIDGE_PORT) $(RT_ARGS)

kinect-test:  ## Windows-side hardware check: print 10 s of Kinect SDK skeleton frames
	powershell.exe -NoProfile -Command "Set-Location -LiteralPath '$(WIN_ROOT)'; python scripts\kinect_bridge_windows.py --print --max-seconds 10"

kinect-diag:  ## Windows-side viewer: depth + player mask + skeletons + tilt (TILT=0 levels the sensor)
	powershell.exe -NoProfile -Command "Set-Location -LiteralPath '$(WIN_ROOT)'; python scripts\kinect_diag_windows.py $(if $(TILT),--tilt $(TILT),)"

kinect-bridge:  ## start the Windows Kinect bridge (Windows Python) streaming to WSL localhost
	powershell.exe -NoProfile -Command "Set-Location -LiteralPath '$(WIN_ROOT)'; python scripts\kinect_bridge_windows.py --host 127.0.0.1 --port $(BRIDGE_PORT)"

live:  ## one command: Windows Kinect bridge + WSL recogniser with OpenCV window (q/ESC quits)
	@test -f "$(CKPT)" || { echo "No checkpoint at $(CKPT)."; exit 1; }
	( sleep 3; powershell.exe -NoProfile -Command "Set-Location -LiteralPath '$(WIN_ROOT)'; python scripts\kinect_bridge_windows.py --host 127.0.0.1 --port $(BRIDGE_PORT) --exit-on-disconnect --give-up-after 20 --print-every 300" ) &
	$(PY) scripts/run_realtime.py --checkpoint "$(CKPT)" --source bridge --port $(BRIDGE_PORT) $(RT_ARGS)
	wait

# ----------------------------------------------------------------------------- notebooks / docker
notebooks:  ## execute all notebooks headlessly into outputs/notebooks (sources stay clean)
	mkdir -p outputs/notebooks
	for nb in notebooks/*.ipynb; do
	  $(PY) -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
	    --ExecutePreprocessor.kernel_name=python3 --output-dir outputs/notebooks "$$nb" || exit 1
	done

docker-build:  ## build the training/evaluation image (no Kinect inside Docker)
	docker build -t kard-skeleton-action .

# ----------------------------------------------------------------------------- cleaning
clean:  ## remove temporary files (caches, __pycache__, *.tmp, notebook checkpoints)
	find . -path ./data/raw -prune -o -type d \( -name __pycache__ -o -name .pytest_cache \
	  -o -name .ruff_cache -o -name .ipynb_checkpoints -o -name '*.egg-info' \) -print -exec rm -rf {} +
	find . -path ./data/raw -prune -o -type f \( -name '*.pyc' -o -name '*.pt.tmp' \) -print -exec rm -f {} +
	rm -rf build dist

clean-cache:  ## remove derived data caches (data/interim, data/processed); raw data untouched
	rm -rf data/interim/* data/processed/*
	touch data/interim/.gitkeep data/processed/.gitkeep

clean-outputs:  ## DELETE all checkpoints/metrics/figures/logs (asks for confirmation)
	@read -p "Delete everything under outputs/ ? [y/N] " a; [ "$$a" = y ] || exit 0
	find outputs -mindepth 1 -not -name .gitkeep -not -path 'outputs/*/.gitkeep' -delete
	mkdir -p outputs/checkpoints outputs/metrics outputs/figures
