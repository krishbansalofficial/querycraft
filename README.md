# Querycraft

Fine tuning Qwen2.5 Coder for text to SQL on BIRD, scored by execution accuracy, with a controlled study of
parameter efficient fine tuning: a LoRA rank sweep, DoRA against LoRA at an equal budget, and full fine tuning
against LoRA, including how much general coding ability each method forgets.

## Problem

Given a natural language question, an optional hint ("external knowledge") and a SQLite database, produce one SQL
query whose result equals the result of the gold query. BIRD databases are large, messy and real (up to hundreds of
columns per database), so the task tests schema understanding as much as SQL syntax.

## Protocol

| Split | Size | Source | Used for |
|---|---|---|---|
| train | see `data/processed/split_stats.json` | BIRD train minus validation databases | training only |
| val | 500 | whole BIRD train databases held out of training | every choice: checkpoints, prompts, DoRA rank |
| dev | 1,534 | BIRD dev (11 unseen databases) | final reporting only, never for selection |
| held back | 100 | written for this project on 3 BIRD dev databases | final reporting, contamination check |

* **Validation is database disjoint from train**, like BIRD dev. Splitting rows instead would let validation share
  schemas with training and overstate how well a checkpoint generalises.
* `tests/test_split.py` fails if any dev or held back database appears in train or validation, or if train and
  validation share a database. `make data` runs the same check.
* **Metric.** A prediction is correct when its result rows equal the gold rows. Comparison ignores row order unless the
  gold query has a top level `ORDER BY`, in which case order must match too, and duplicates count. The official
  BIRD criterion (`set(pred) == set(gold)`) is reported alongside for leaderboard comparability.
* **Error categories** for wrong predictions: syntax, wrong table, wrong column, wrong join, wrong aggregation,
  timeout, other. Execution errors are tagged from the SQLite message. Queries that run but return the wrong rows
  are tagged by a structural diff against gold (tables, then join tables and ON columns, then aggregates and
  GROUP BY, then columns). "Other" means the right skeleton with a wrong literal, filter or ordering.
* **Forgetting probe.** HumanEval (164) plus MBPP sanitized test (257), pass@1, greedy. The base model is a Coder
  model, so general Python coding is the ability that SQL specialisation is most likely to erode. It is also scored
  by execution, like the main metric. Two suites give 421 problems instead of 164, which narrows the confidence
  interval enough to separate methods.
* **Contamination check.** BIRD dev has been public since 2023 and may sit in pretraining data. The held back set
  asks new questions over the same three dev databases (superhero, student_club, formula_1). The report compares
  each model's held back accuracy with its dev accuracy on those same databases.

## Method

* **Prompt.** System instruction, then the schema as `CREATE TABLE` statements with key constraints and three
  example values per column, then the hint, then the question. The assistant turn is the bare SQL.
* **Schema pruning.** If the schema exceeds 1,280 tokens, tables and then columns are dropped by lexical relevance to
  the question and hint. Key columns are never dropped. Lengths before and after pruning, and the share of examples
  whose gold tables all survive, are in `data/processed/prompt_stats.json` and below.
* **Training.** TRL SFT with loss on the SQL only, 560 optimizer steps at an effective batch of 16 (about one epoch),
  cosine schedule with 3% warmup, seed 0. Every trained run sees the same examples in the same order and the same
  number of steps. LoRA family runs use lr 2e-4, alpha = 2 x rank, dropout 0.05, all linear projections. Full fine
  tuning uses lr 1e-5 with fp32 master weights (bf16 weights would round away updates of that size).
* **Selection.** Checkpoints at steps 280, 420 and 560 are scored on validation; the best one goes to dev. DoRA
  uses the rank that won the QLoRA sweep on validation.
* **Inference.** vLLM, greedy, bf16 base for every model so precision is constant across comparisons. LoRA
  checkpoints are served as adapters; DoRA checkpoints are merged first because vLLM does not serve DoRA adapters.

## Results

<!-- RESULTS:START -->
### Main results (BIRD dev, execution accuracy %)

| Model | Val EX (selection) | Dev EX [95% CI] | Dev EX (BIRD official) | Simple | Moderate | Challenging | Held back EX | Dev EX on held back DBs | HumanEval | MBPP | Trained on | Train time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5B zero shot | 24.0 | 27.44 [25.27, 29.73] | 30.9 | 34.49 | 16.59 | 17.24 | 61.0 | 32.32 | 72.56 | 65.37 |  |  |
| 1.5B few shot (3) | 21.4 | 25.95 [23.81, 28.2] | 28.29 | 32.11 | 16.59 | 16.55 | 49.0 | 29.72 |  |  |  |  |
| 1.5B QLoRA r16 | 32.2 | 42.31 [39.86, 44.8] | 47.59 | 50.05 | 32.54 | 24.14 | 66.0 | 52.49 | 68.9 | 68.09 | NVIDIA GeForce RTX 4070 Laptop GPU x1 | 2.12 h |
| 1.5B LoRA (bf16) r16 | 33.6 | 43.94 [41.47, 46.43] | 49.09 | 51.89 | 32.97 | 28.28 | 68.0 | 54.66 | 68.9 | 68.09 | NVIDIA GeForce RTX 4070 Laptop GPU x1 | 1.50 h |

Held back EX vs dev EX on the same three databases is the contamination check: a model that memorised BIRD dev would score clearly higher on dev questions than on new questions over the same schemas.

### Error categories on BIRD dev (count of wrong predictions)

| Model | syntax | wrong_table | wrong_column | wrong_join | wrong_aggregation | timeout | other |
|---|---|---|---|---|---|---|---|
| 1.5B zero shot | 26 | 192 | 626 | 111 | 56 | 1 | 101 |
| 1.5B few shot (3) | 30 | 187 | 660 | 94 | 61 | 2 | 102 |
| 1.5B QLoRA r16 | 1 | 243 | 307 | 145 | 78 | 0 | 111 |
| 1.5B LoRA (bf16) r16 | 3 | 241 | 268 | 155 | 75 | 0 | 118 |

### Checkpoint selection on validation (EX %, chosen in bold)

| Model | checkpoint-280 | checkpoint-420 | checkpoint-560 | Trainable params | Peak VRAM (GB) |
|---|---|---|---|---|---|
| 1.5B QLoRA r16 | 31.6 | 32.2 | **32.2** | 18464768 | 2.37 |
| 1.5B LoRA (bf16) r16 | 33.2 | **33.6** | 33.2 | 18464768 | 3.68 |

### Targets (not yet measured)

These runs are configured but have no logged results yet. No numbers are reported for them.

* 7B zero shot (`configs/rented/01_qwen7b_zeroshot.yaml`, A100 80GB)
* 7B few shot (3) (`configs/rented/02_qwen7b_fewshot.yaml`, A100 80GB)
* 7B QLoRA r8 (`configs/rented/10_qwen7b_qlora_r8.yaml`, A100 80GB)
* 7B QLoRA r16 (`configs/rented/11_qwen7b_qlora_r16.yaml`, A100 80GB)
* 7B QLoRA r64 (`configs/rented/12_qwen7b_qlora_r64.yaml`, A100 80GB)
* 7B DoRA (4 bit base) at best sweep rank (`configs/rented/20_qwen7b_dora_best.yaml`, A100 80GB)
* 1.5B full fine tune (`configs/rented/30_qwen1.5b_full.yaml`, A100 80GB)

![dev_ex_by_difficulty.png](docs/figures/dev_ex_by_difficulty.png)
![learn_vs_forget.png](docs/figures/learn_vs_forget.png)
<!-- RESULTS:END -->

## Hardware

| Runs | Hardware |
|---|---|
| data, tests, 1.5B baselines, 1.5B QLoRA and LoRA, smoke test | Lenovo Legion Pro 5, RTX 4070 Laptop 8 GB, WSL2 Ubuntu 24.04, driver 560.94 |
| 7B baselines, 7B rank sweep, 7B DoRA, 1.5B full fine tune | 1x A100 80GB (RunPod), one scripted session |

The exact GPU, package versions, git commit, seed, config and wall clock of every stage are written to
`runs/<name>/<stage>_meta.json`.

## Reproduce

```bash
make env                      # venv with torch 2.13 cu129 + vLLM 0.32 cu129, then a GPU check
make all PROFILE=local        # data, tests, 1.5B baselines, 1.5B QLoRA + LoRA, report
make smoke                    # rehearse the rented session on Qwen2.5 Coder 0.5B with 20 examples
# on the A100 (network volume at /workspace):
bash scripts/runpod_session.sh   # every 7B job + 1.5B full fine tune, then stops the pod
# back on the laptop, after copying /workspace/querycraft_sync/runs/* into runs/:
make report
```

Individual stages: `python -m querycraft.train <config>`, `python -m querycraft.run_eval <config>`,
`python -m querycraft.evaluate <predictions.jsonl>`.

## Repository

```
querycraft/   data.py (splits, leakage check)  schema.py (serializer, pruning)  prompts.py
              train.py (training only)  infer.py (vLLM inference only)  evaluate.py + execution.py (scoring only)
              run_eval.py (selection on val, then dev)  forgetting.py  report.py
configs/      base.yaml, train.yaml, local/, rented/, smoke/
data/heldback the 100 project written questions (draft.py validates every gold query)
tests/        harness, split + leakage, schema
```

## What did not work

* **First pruning design.** Dropping whole tables by lexical relevance while keeping every kept table in full detail
  (types and example values) lost a gold table in 38.6% of validation prompts and 15.1% of dev prompts at a 1,280 token
  budget. Showing low relevance tables in compact form (names and keys only) before dropping anything, plus a score
  boost for tables one foreign key away from a relevant table, raised validation gold table recall to 96.2% at the same
  budget. A 1,536 token budget only added 1.4 points on validation, so 1,280 stayed.
* **Row level validation split.** Rejected before any run: validation rows from training databases would share schemas
  with training, unlike BIRD dev, and would reward schema memorisation when picking checkpoints.
* **Latest vLLM on the laptop.** vLLM 0.33 needs torch 2.14 built for CUDA 13, which driver 560 (CUDA 12.6) cannot
  run. The stack is pinned to vLLM 0.32 and torch 2.13 cu129 builds, which run through CUDA minor version
  compatibility. Two more fixes were needed: removing a CUDA 13 `torchcodec` build that crashed vLLM at import, and
  installing `python3-dev`, because Triton compiles a C helper at runtime.
* **Full fine tuning on 8 GB.** 1.5B parameters with fp32 master weights, fp32 gradients and AdamW states need about
  24 GB before activations. Even bf16 weights with 8 bit AdamW need about 9 GB. Full fine tuning was moved to the A100;
  its LoRA partner trains on the laptop with the identical recipe.
* **FlashInfer sampler on a driver only machine.** vLLM 0.32 compiles a FlashInfer top k / top p sampling kernel at
  startup, which needs `nvcc`. WSL and slim pod images ship only the driver, so the first baseline crashed. Greedy
  decoding needs no such kernel, so `VLLM_USE_FLASHINFER_SAMPLER=0` is set before vLLM is imported.
* **BIRD archive layout.** The train archive ships macOS `__MACOSX` resource fork files that mirror every database
  path. A naive path search picked them first, so validation pointed at non databases. The loader now skips them.
