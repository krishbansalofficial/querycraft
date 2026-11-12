# One command reproduces everything:   make all PROFILE=local   (laptop)   |   make all PROFILE=rented   (A100)
# Config files starting with 0 are baselines (no training); the rest are trained in filename order.
PROFILE ?= local
PY ?= .venv/bin/python
CFG_DIR := configs/$(PROFILE)
BASELINES := $(sort $(wildcard $(CFG_DIR)/0*.yaml))
TRAINED := $(sort $(filter-out $(BASELINES),$(wildcard $(CFG_DIR)/*.yaml)))
export PYTHONPATH := $(CURDIR)

.PHONY: env data baseline train eval report test all smoke audit

env:
	bash scripts/setup_env.sh
	$(PY) scripts/verify_env.py

data:
	bash scripts/download_bird.sh
	$(PY) -m querycraft.data
	$(PY) -m querycraft.prompt_stats configs/local/01_qwen1.5b_zeroshot.yaml

baseline:
	@for c in $(BASELINES); do echo "== baseline $$c"; $(PY) -m querycraft.run_eval $$c || exit 1; done

# train also evaluates each run: DoRA picks its rank from the sweep's validation scores
train:
	@for c in $(TRAINED); do echo "== train $$c"; $(PY) -m querycraft.train $$c && $(PY) -m querycraft.run_eval $$c || exit 1; done

eval:
	@for c in $(TRAINED); do echo "== eval $$c"; $(PY) -m querycraft.run_eval $$c || exit 1; done

report:
	$(PY) -m querycraft.report

test:
	$(PY) -m pytest -q tests

all: data test baseline train eval report

# local rehearsal of the rented session on Qwen2.5 Coder 0.5B with 20 examples
smoke:
	RUNS_DIR=runs_smoke SHUTDOWN=0 PROFILE=smoke bash scripts/runpod_session.sh

audit:
	 -m querycraft.audit runs
