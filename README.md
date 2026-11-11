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
Run `make report` to fill this section from logged runs.
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

<!-- filled from the runs -->
