# v2 compute plan (GPU audit, 2026-11-23)

Budget: $0. Sources: measured wall clock and peak VRAM in `runs/*/*_meta.json` and the README tables;
`nvidia-smi` on the laptop (RTX 4070 Laptop, 8,188 MiB, driver 560.94, CUDA 12.6, 0 MiB in use at idle);
WSL has 15 GB RAM, 32 threads, 854 GB free disk.

## Measured anchors (from runs/)

| What | Hardware | Wall clock | Peak VRAM |
|---|---|---|---|
| 1.5B LoRA r16 training, 560 steps | laptop | 1.50 h | 3.68 GB |
| 1.5B table selector training (same recipe) | laptop | 1.44 h | 3.63 GB |
| 1.5B second pass, val + dev + held back (2,134 prompts) | laptop | 186 to 239 s | vLLM at 0.85 of 8 GB |
| 1.5B 8 samples x 3 temperatures, val + dev + held back | laptop | 1,166 s | vLLM |
| GRPO training | laptop | 4.07 h | 6.62 GB |
| 7B DoRA eval incl. merge, 3 val checkpoints, dev, held back, forgetting | A100 80GB | 1,091 s | n/a |
| 7B zero shot eval, all splits + forgetting | A100 80GB | 451 s | n/a |

The 7B DoRA adapter is only on the Modal volume (`runs/qwen7b_dora_best/ckpt/checkpoint-560`); the laptop has
its dev predictions but not the weights. BIRD databases needed outside the laptop: dev 1.4 GB, val 280 MB.

## Every GPU job v2 needs

| # | Job | Model | Est. VRAM | Est. GPU h | Where |
|---|---|---|---|---|---|
| 1 | Bi encoder fine tune (in batch + same database hard negatives) | bge-small-en-v1.5, 33M | 3 to 4 GB | 0.5 | laptop |
| 2 | Cross encoder reranker fine tune | bge-reranker-base, 278M (MiniLM 33M if slow) | 5 to 6 GB | 1.0 | laptop |
| 3 | Embed every table, score val/dev/held back, k and hop sweep on val | 33M + 278M | < 2 GB | 0.2 | laptop |
| 4 | 1.5B SFT second pass over retrieved tables (val sweep of 3 to 4 settings, then dev + held back) | 1.5B bf16 | ~7 GB (vLLM 0.85) | 0.4 | laptop |
| 5 | Abstain features: 8 samples on the retrieval system for val + dev (old vote files reused for the baseline) | 1.5B bf16 | ~7 GB | 0.4 | laptop |
| 6 | Agent loop, 1.5B, val + dev + held back (step synchronous batching, step limit 6) | 1.5B bf16, context 8k | ~7 GB | 2.0 | laptop |
| 7 | AWQ 4 bit quantization of merged 1.5B SFT (llm-compressor W4A16) | 1.5B | ~5 GB | 0.3 | laptop |
| 8 | Bench 1.5B: dev EX bf16 vs AWQ, p50/p99, throughput at concurrency 1/8/32, peak VRAM | 1.5B | 2 to 7 GB | 1.5 | laptop |
| 9 | Eval gate baseline: 200 question val slice | 1.5B | ~7 GB | 0.1 | laptop (CI itself is CPU: replays cached predictions) |
| 10 | Iowa Liquor demo: 30 questions, single pass + agent | 1.5B | ~7 GB | 0.2 | laptop |
| 11 | Merge 7B DoRA into fp16 base, smoke test vLLM on T4 | 7B fp16, tensor parallel 2 | ~12 GB per T4 | 0.5 | Kaggle |
| 12 | 7B single pass on T4 fp16, val + dev + held back (the agent's paired baseline on the same hardware and precision) | 7B fp16 TP2 | ~12 GB per T4 | 1.0 | Kaggle |
| 13 | 7B agent loop, val + dev + held back | 7B fp16 TP2 | ~13 GB per T4 | 5.0 | Kaggle |
| 14 | AWQ 4 bit quantization of merged 7B DoRA | 7B, sequential onloading | ~10 GB GPU, ~20 GB RAM | 1.0 | Kaggle |
| 15 | 7B AWQ bench + dev EX on the laptop (paired against the existing A100 bf16 dev predictions) | 7B W4A16, ~5.6 GB weights | ~7.4 GB, ~1 GB KV | 0.7 | laptop, after a 10 min fit probe |
| 16 | 7B oracle + self linking (already scheduled, v1) | 7B DoRA | A100 | 0.5 | Modal cron, 12:00 UTC Dec 1 |

Not GPU: XGBoost/LightGBM abstain model, sqlglot features, FAISS, MLflow, drift report, FastAPI, Prometheus,
Grafana, OpenTelemetry, CI gate, Iowa download. Hour estimates for jobs 11 to 15 assume a T4 is 6 to 8 times
slower than the A100 at 7B; they are estimates, not measurements.

## What fits on the laptop

Jobs 1 to 10 fit: every one is at or below the 1.5B evaluation footprint that already ran there (vLLM at 0.85
of 8 GB) or the 6.62 GB GRPO peak. Job 15 probably fits (5.6 GB of 4 bit weights leaves about 1 GB of KV cache,
about 19k tokens at 57 KB per token, so 4 or 5 concurrent 4k sequences); a 10 minute probe decides, and if it
does not fit the 7B bench rows are TBD. 7B in bf16 or fp16 does not fit (15 GB of weights), which the README
already records.

## Free provider per off laptop job

| Provider | Limits | Status of the numbers | Use |
|---|---|---|---|
| Kaggle | 2x T4 16 GB, ~30 GPU h/week, 12 h sessions (some docs say 9 h), phone verification for GPU and internet | **unverified**: kaggle.com/docs pages render empty without JavaScript | jobs 11 to 14: the only free option with 32 GB of GPU memory, enough for 7B fp16 split over two T4s |
| Modal Starter | $30/month credit; A100 80GB $0.000694/s (~$2.50/h), A100 40GB ~$2.10/h, L40S ~$1.95/h, L4 ~$0.80/h, T4 ~$0.59/h | verified on modal.com/pricing today | fallback only. November credit is spent ($1.22 left): nothing before Dec 1 |
| Colab free | T4, max 12 h, limits unpublished, "not guaranteed" | verified on the Colab FAQ (no GPU hour numbers exist) | backup for job 14 only if Kaggle quota runs out; one T4 cannot hold 7B fp16, so not for 11 to 13 |

Why Kaggle over Modal for 7B: same answer at $0 with no credit risk, and the December credit stays free for a
fallback. The T4 has no bf16, so 7B runs in fp16; the agent is compared against a single pass run on the same
T4s in fp16 (job 12), never against the A100 bf16 numbers.

## Schedule

| When | Jobs | Provider hours |
|---|---|---|
| Day 1 to 2 after go | 1 to 10 on the laptop, notebooks for 11 to 14 written | laptop ~6.6 h |
| This week (when you run them) | Notebook A: 11 + 12 (~1.5 h). Notebook B: 13 (~5 h, own session). Notebook C: 14 (~1 h) | Kaggle ~7.5 h of ~30 |
| After C's output is back | 15 on the laptop | laptop ~0.7 h |
| Dec 1 | 16 (already armed, worst case $1.50 at the ledger's $3/h) | Modal ~0.5 h |
| Only if vLLM fails on T4 | 12 + 13 on Modal A100 in December, under `budget.py`, ledger logged | Modal ~1.5 h, ~$4.50 |

Totals: laptop ~7.3 h, Kaggle ~7.5 h, Modal 0.5 h planned (2.0 h worst case, $6 of the December $30), Colab 0 h.

## What cannot be done free, and the cut

Nothing has to be paid for. The risks and the cuts that keep it at $0:

* **vLLM on T4.** vLLM 0.32's support for compute capability 7.5 is untested here. Notebook A runs a 5 minute
  smoke test first. If it fails: run 12 and 13 on the Modal A100 after Dec 1 (fits the cap). If that is also not
  possible: drop the 7B agent rows and report 7B as TBD.
* **Kaggle session length.** If sessions are 9 h, not 12, job 13 still fits (~5 h); it writes results per
  question and resumes from them.
* **7B AWQ on the laptop.** If the probe shows it does not fit, the 7B bench rows are TBD and only 1.5B is
  benchmarked, as the brief allows.
* **Deployment.** Local only (docker compose on the laptop). AWS and Modal deploy scripts are written, never run.

## Setup only you can do

1. **Kaggle**: sign in, then Settings > Phone verification (needed for GPU and internet in notebooks).
2. **Kaggle API token**: Settings > API > Create New Token, then in WSL:
   `mkdir -p ~/.kaggle && cp /mnt/c/Users/<you>/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`.
   With that, I upload two private datasets myself: `querycraft-bird-eval` (dev + val databases, split files,
   ~1.7 GB) and `querycraft-7b-dora` (the adapter, fetched with `modal volume get`, a read, no GPU).
3. **Nothing new for Modal**: the token is already in `~/.modal.toml` and the workspace spend limit is $0. Keep
   it at $0.
4. **Colab**: nothing unless Kaggle quota runs out.
5. **Docker Desktop**: already installed. Turn on Settings > Resources > WSL integration for Ubuntu so
   `docker compose` works from WSL.
