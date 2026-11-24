# One command reproduces everything:   make all PROFILE=local   (laptop)   |   make all PROFILE=rented   (A100)
# Config files starting with 0 are baselines (no training); the rest are trained in filename order.
PROFILE ?= local
PY ?= .venv/bin/python
CFG_DIR := configs/$(PROFILE)
BASELINES := $(sort $(wildcard $(CFG_DIR)/0*.yaml))
TRAINED := $(sort $(filter-out $(BASELINES),$(wildcard $(CFG_DIR)/*.yaml)))
export PYTHONPATH := $(CURDIR)

.PHONY: env data baseline train eval report test all smoke audit retriever abstain agent serve bench gate gate-refresh warehouse drift track notebooks

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
	$(PY) -m querycraft.audit runs

# ---------------------------------------------------------------- v2 (docs/plans/v2.md); every choice made on val
RCFG := configs/retriever/table_retriever.yaml

retriever:      # bi + cross encoder on train, rank tables, pick source/k/hop on val, second pass (laptop GPU)
	$(PY) -m querycraft.retriever train $(RCFG)
	$(PY) -m querycraft.retriever rank $(RCFG)
	$(PY) -m querycraft.retriever sweep $(RCFG)
	$(PY) -m querycraft.retriever link $(RCFG)
	$(PY) -m querycraft.run_eval configs/retriever/70_qwen1.5b_lora_r16_retrieved.yaml

abstain:        # learned answer or abstain model (CPU; needs the v1 confidence run and the retriever)
	$(PY) -m querycraft.abstain configs/agent/abstain.yaml

agent:          # agent loop over the 1.5B SFT model: tune on val, then val/dev/held back/iowa (laptop GPU)
	$(PY) -m querycraft.agent configs/agent/agent_1.5b.yaml

warehouse:      # Iowa Liquor Sales into SQLite (row cap for a laptop) + validate the 30 demo questions
	$(PY) scripts/load_iowa_liquor.py --max-rows 2000000
	$(PY) scripts/iowa_questions.py

serve:          # vLLM with the LoRA adapter on :8001, then the API on :8000 (Ctrl+C stops both)
	VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct --port 8001 --enable-lora 	  --max-lora-rank 64 --lora-modules querycraft=runs/qwen1.5b_lora_r16/ckpt/checkpoint-420 --max-model-len 4096 	  --gpu-memory-utilization 0.80 & sleep 90; .venv/bin/uvicorn serve.app:app --port 8000; kill %1

bench:          # AWQ quantization (separate venv), AWQ accuracy, then latency/throughput of bf16 vs AWQ
	[ -f runs/qwen1.5b_lora_r16/awq/config.json ] || .venv-quant/bin/python serve/bench/quantize_awq.py runs/qwen1.5b_lora_r16/grpo/sft_merged runs/qwen1.5b_lora_r16/awq
	$(PY) -m querycraft.run_eval configs/serve/81_qwen1.5b_lora_r16_awq.yaml
	$(PY) -m querycraft.bench configs/serve/bench.yaml

gate:           # CI evaluation gate: replay cached completions on the 200 question val slice (CPU)
	$(PY) -m querycraft.gate check

gate-refresh:   # GPU + BIRD databases: regenerate the gate fixtures and baseline, then commit data/gate/
	$(PY) -m querycraft.gate refresh configs/retriever/70_qwen1.5b_lora_r16_retrieved.yaml

drift:
	$(PY) -m querycraft.drift

track:          # mirror runs/ into MLflow (sqlite:///mlflow/mlflow.db)
	$(PY) -m querycraft.tracking

notebooks:
	$(PY) notebooks/make_notebooks.py
