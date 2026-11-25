### Main results (BIRD dev, execution accuracy %)

| Model | Val EX (selection) | Dev EX [95% CI] | Dev EX (BIRD official) | Simple | Moderate | Challenging | Held back EX | Dev EX on held back DBs | HumanEval | MBPP | Trained on | Train time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.5B zero shot | 24.0 | 27.44 [25.27, 29.73] | 30.9 | 34.49 | 16.59 | 17.24 | 61.0 | 32.32 | 72.56 | 65.37 |  |  |
| 1.5B few shot (3) | 21.4 | 25.95 [23.81, 28.2] | 28.29 | 32.11 | 16.59 | 16.55 | 49.0 | 29.72 |  |  |  |  |
| 1.5B QLoRA r16 | 32.2 | 42.31 [39.86, 44.8] | 47.59 | 50.05 | 32.54 | 24.14 | 66.0 | 52.49 | 68.9 | 68.09 | NVIDIA GeForce RTX 4070 Laptop GPU x1 | 2.12 h |
| 1.5B LoRA (bf16) r16 | 33.6 | 43.94 [41.47, 46.43] | 49.09 | 51.89 | 32.97 | 28.28 | 68.0 | 54.66 | 68.9 | 68.09 | NVIDIA GeForce RTX 4070 Laptop GPU x1 | 1.50 h |
| 7B zero shot | 53.2 | 51.43 [48.93, 53.93] | 56.26 | 58.16 | 43.97 | 32.41 | 88.0 | 62.91 | 87.8 | 85.6 |  |  |
| 7B few shot (3) | 50.4 | 50.13 [47.63, 52.63] | 55.74 | 57.62 | 40.73 | 32.41 | 90.0 | 60.74 |  |  |  |  |
| 7B QLoRA r8 | 56.4 | 51.69 [49.19, 54.19] | 57.63 | 59.03 | 43.75 | 30.34 | 82.0 | 62.47 | 84.76 | 84.44 | NVIDIA A100-SXM4-80GB x1 | 1.45 h |
| 7B QLoRA r16 | 56.0 | 52.15 [49.65, 54.64] | 57.95 | 59.35 | 44.83 | 29.66 | 84.0 | 63.77 | 84.15 | 83.66 | NVIDIA A100-SXM4-80GB x1 | 1.46 h |
| 7B QLoRA r64 | 56.0 | 52.67 [50.17, 55.16] | 58.15 | 59.14 | 45.91 | 33.1 | 83.0 | 62.47 | 83.54 | 82.88 | NVIDIA A100-SXM4-80GB x1 | 1.47 h |
| 7B DoRA (4 bit base) r8 | 56.4 | 51.69 [49.19, 54.19] | 57.76 | 59.14 | 43.32 | 31.03 | 83.0 | 62.69 | 85.98 | 84.44 | NVIDIA A100-SXM4-80GB x1 | 1.09 h |
| 1.5B full fine tune | 36.2 | 42.76 [40.31, 45.25] | 47.52 | 49.95 | 34.27 | 24.14 | 65.0 | 54.01 | 34.15 | 52.14 | NVIDIA A100-SXM4-80GB x1 | 0.48 h |
| 1.5B SFT LoRA + GRPO (binary reward) | 35.2 | 44.52 [42.05, 47.02] | 48.83 | 52.11 | 34.27 | 28.97 | 73.0 | 55.75 | 71.34 | 68.09 | NVIDIA GeForce RTX 4070 Laptop GPU x1 | 4.07 h |
| 1.5B SFT LoRA + GRPO (shaped reward) | 35.8 | 44.07 [41.6, 46.56] | 48.04 | 51.78 | 33.41 | 28.97 | 73.0 | 54.66 | 68.29 | 67.7 | NVIDIA GeForce RTX 4070 Laptop GPU x1 | 14.62 h |
| 1.5B SFT LoRA, oracle tables (upper bound) | 45.0 | 48.5 [46.01, 51.0] | 54.17 | 58.49 | 35.34 | 26.9 | 79.0 | 61.61 |  |  |  |  |
| 1.5B SFT LoRA, self linked tables (2 pass) | 36.6 | 44.46 [41.99, 46.96] | 49.54 | 52.43 | 33.19 | 29.66 | 76.0 | 54.45 |  |  |  |  |
| 1.5B SFT LoRA, trained table selector (2 pass) | 34.4 | 44.72 [42.25, 47.22] | 49.87 | 54.49 | 31.03 | 26.21 | 63.0 | 55.75 |  |  |  |  |
| 1.5B SFT LoRA, trained table retriever (2 pass) | 35.6 | 43.22 [40.76, 45.71] | 48.24 | 51.57 | 32.11 | 25.52 | 68.0 | 54.01 |  |  |  |  |

Held back EX vs dev EX on the same three databases is the contamination check: a model that memorised BIRD dev would score clearly higher on dev questions than on new questions over the same schemas.

### Error categories on BIRD dev (count of wrong predictions)

| Model | syntax | wrong_table | wrong_column | wrong_join | wrong_aggregation | timeout | other |
|---|---|---|---|---|---|---|---|
| 1.5B zero shot | 26 | 192 | 626 | 111 | 56 | 1 | 101 |
| 1.5B few shot (3) | 30 | 187 | 660 | 94 | 61 | 2 | 102 |
| 1.5B QLoRA r16 | 1 | 243 | 307 | 145 | 78 | 0 | 111 |
| 1.5B LoRA (bf16) r16 | 3 | 241 | 268 | 155 | 75 | 0 | 118 |
| 7B zero shot | 5 | 189 | 202 | 142 | 75 | 1 | 131 |
| 7B few shot (3) | 7 | 190 | 217 | 141 | 74 | 2 | 134 |
| 7B QLoRA r8 | 1 | 192 | 162 | 170 | 74 | 0 | 142 |
| 7B QLoRA r16 | 1 | 203 | 161 | 155 | 74 | 0 | 140 |
| 7B QLoRA r64 | 2 | 197 | 157 | 153 | 76 | 0 | 141 |
| 7B DoRA (4 bit base) r8 | 1 | 201 | 155 | 164 | 77 | 0 | 143 |
| 1.5B full fine tune | 8 | 244 | 247 | 167 | 83 | 1 | 128 |
| 1.5B SFT LoRA + GRPO (binary reward) | 3 | 263 | 239 | 166 | 71 | 0 | 109 |
| 1.5B SFT LoRA + GRPO (shaped reward) | 2 | 235 | 276 | 165 | 68 | 0 | 112 |
| 1.5B SFT LoRA, oracle tables (upper bound) | 3 | 116 | 241 | 185 | 90 | 0 | 155 |
| 1.5B SFT LoRA, self linked tables (2 pass) | 4 | 244 | 254 | 148 | 79 | 0 | 123 |
| 1.5B SFT LoRA, trained table selector (2 pass) | 10 | 250 | 241 | 131 | 81 | 0 | 135 |
| 1.5B SFT LoRA, trained table retriever (2 pass) | 6 | 263 | 258 | 143 | 80 | 0 | 121 |

### Checkpoint selection on validation (EX %, chosen in bold)

| Model | checkpoint-280 | checkpoint-420 | checkpoint-560 | Trainable params | Peak VRAM (GB) |
|---|---|---|---|---|---|
| 1.5B QLoRA r16 | 31.6 | 32.2 | **32.2** | 18464768 | 2.37 |
| 1.5B LoRA (bf16) r16 | 33.2 | **33.6** | 33.2 | 18464768 | 3.68 |
| 7B QLoRA r8 | 54.6 | 56.0 | **56.4** | 20185088 | 13.72 |
| 7B QLoRA r16 | 54.0 | 55.4 | **56.0** | 40370176 | 13.83 |
| 7B QLoRA r64 | 52.0 | 54.6 | **56.0** | 161480704 | 14.74 |
| 7B DoRA (4 bit base) r8 | 54.4 | 54.6 | **56.4** | 21575680 | 16.93 |
| 1.5B full fine tune | 33.4 | 34.0 | **36.2** | all | 26.95 |

| Model | checkpoint-25 | checkpoint-50 | checkpoint-75 | checkpoint-100 | checkpoint-125 | Trainable params | Peak VRAM (GB) |
|---|---|---|---|---|---|---|---|
| 1.5B SFT LoRA + GRPO (binary reward) | 32.8 | 33.4 | 34.0 | 34.0 | **35.2** | 18464768 | 6.62 |
| 1.5B SFT LoRA + GRPO (shaped reward) | 32.8 | 34.0 | 34.0 | 34.6 | **35.8** | 18464768 | 6.48 |

### Does fine tuning beat zero shot? Paired against the same size zero shot model

p is an exact two sided sign test on the questions where exactly one of the two models is right (dev: 1,534 questions, held back: 100).

| Model | Zero shot dev EX | Dev EX | Only FT right | Only zero shot right | p | Zero shot held back | Held back | Only FT right | Only zero shot right | p |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5B QLoRA r16 | 27.44 | 42.31 | 300 | 72 | <0.001 | 61.0 | 66.0 | 14 | 9 | 0.405 |
| 1.5B LoRA (bf16) r16 | 27.44 | 43.94 | 320 | 67 | <0.001 | 61.0 | 68.0 | 15 | 8 | 0.21 |
| 7B QLoRA r8 | 51.43 | 51.69 | 115 | 111 | 0.842 | 88.0 | 82.0 | 4 | 10 | 0.18 |
| 7B QLoRA r16 | 51.43 | 52.15 | 119 | 108 | 0.507 | 88.0 | 84.0 | 4 | 8 | 0.388 |
| 7B QLoRA r64 | 51.43 | 52.67 | 121 | 102 | 0.228 | 88.0 | 83.0 | 5 | 10 | 0.302 |
| 7B DoRA (4 bit base) r8 | 51.43 | 51.69 | 114 | 110 | 0.841 | 88.0 | 83.0 | 4 | 9 | 0.267 |
| 1.5B full fine tune | 27.44 | 42.76 | 323 | 88 | <0.001 | 61.0 | 65.0 | 19 | 15 | 0.608 |

### Built on an SFT model, paired against it

GRPO arms start from the 1.5B LoRA r16 SFT checkpoint and train 125 steps on 1,000 prompts the SFT model solves 1 to 7 times out of 8. Schema linking runs use an SFT model unchanged (1.5B LoRA r16, or the 7B DoRA run) and only change which tables the prompt shows (`docs/plans/schema_linking.md`); the oracle row uses the gold SQL's tables and is an upper bound, not a usable system. p is an exact two sided sign test on the questions where exactly one of the two models is right.

| Model | SFT dev EX | Dev EX | Only new right | Only SFT right | p | SFT held back | Held back | Only new right | Only SFT right | p |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5B SFT LoRA + GRPO (binary reward) | 43.94 | 44.52 | 73 | 64 | 0.494 | 68.0 | 73.0 | 10 | 5 | 0.302 |
| 1.5B SFT LoRA + GRPO (shaped reward) | 43.94 | 44.07 | 81 | 79 | 0.937 | 68.0 | 73.0 | 7 | 2 | 0.18 |
| 1.5B SFT LoRA, oracle tables (upper bound) | 43.94 | 48.5 | 134 | 64 | <0.001 | 68.0 | 79.0 | 14 | 3 | 0.013 |
| 1.5B SFT LoRA, self linked tables (2 pass) | 43.94 | 44.46 | 39 | 31 | 0.403 | 68.0 | 76.0 | 8 | 0 | 0.008 |
| 1.5B SFT LoRA, trained table selector (2 pass) | 43.94 | 44.72 | 97 | 85 | 0.415 | 68.0 | 63.0 | 4 | 9 | 0.267 |
| 1.5B SFT LoRA, trained table retriever (2 pass) | 43.94 | 43.22 | 40 | 51 | 0.294 | 68.0 | 68.0 | 3 | 3 | 1.0 |

### Knowing when it is wrong: voting and abstention (BIRD dev)

8 sampled queries per question are executed and grouped by result; the largest group is the answer and its share is the confidence. Temperature and thresholds are chosen on val; the threshold columns answer only questions at or above the val threshold that kept 50% or 70% of val questions.

| Model | Single answer EX | Vote of 8 EX | Confidence AUROC | Answers at val 50% threshold | EX there | Answers at val 70% threshold | EX there | Held back EX at val 50% threshold |
|---|---|---|---|---|---|---|---|---|
| 1.5B zero shot | 27.44 | 37.38 | 0.7693 | 56.23% | 54.29 | 84.67% | 43.68 | 86.3 (73.0% answered) |
| 1.5B QLoRA r16 | 42.31 | 45.6 | 0.7494 | 56.03% | 61.23 | 85.91% | 51.03 | 87.14 (70.0% answered) |
| 1.5B LoRA (bf16) r16 | 43.94 | 48.53 | 0.7411 | 56.49% | 64.09 | 81.41% | 55.29 | 90.77 (65.0% answered) |

### Picking the tables: set metrics (v2 part 1)

Gold tables come from the gold SQL. "Every gold table kept" is the share of questions whose prompt can contain everything the answer needs. Self linking and the selector are the v1 rows; the retriever's source, k and foreign key hop were chosen by second pass val EX (chosen: **bi_k2_hop**).

| Method | Split | Recall | Precision | Every gold table kept | Mean tables shown |
|---|---|---|---|---|---|
| Self linking (first pass tables + 1 FK hop) | val | 0.9776 | 0.2057 | 0.962 | 9.34 |
| Trained table selector (lists tables) | val | 0.8271 | 0.8469 | 0.71 | 1.92 |
| Retriever, bi encoder, top 2 | val | 0.6399 | 0.629 | 0.41 | 2.0 |
| **Retriever, bi encoder, top 2 + 1 FK hop** | val | 0.939 | 0.2531 | 0.898 | 7.29 |
| Retriever, bi encoder, top 3 | val | 0.7386 | 0.484 | 0.55 | 3.0 |
| Retriever, bi encoder, top 4 | val | 0.8098 | 0.398 | 0.652 | 4.0 |
| Retriever, bi encoder, top 5 | val | 0.8505 | 0.3344 | 0.726 | 5.0 |
| Retriever, bi encoder + cross encoder rerank, top 2 | val | 0.767 | 0.754 | 0.622 | 2.0 |
| Retriever, bi encoder + cross encoder rerank, top 3 | val | 0.8545 | 0.56 | 0.734 | 3.0 |
| Retriever, bi encoder + cross encoder rerank, top 4 | val | 0.9034 | 0.444 | 0.82 | 4.0 |
| Retriever, bi encoder + cross encoder rerank, top 5 | val | 0.9318 | 0.3664 | 0.872 | 5.0 |
| Self linking (first pass tables + 1 FK hop) | dev | 0.9833 | 0.3771 | 0.9739 | 5.08 |
| Trained table selector (lists tables) | dev | 0.889 | 0.9078 | 0.8227 | 1.91 |
| Retriever, bi encoder, top 2 | dev | 0.7432 | 0.7246 | 0.5828 | 2.0 |
| **Retriever, bi encoder, top 2 + 1 FK hop** | dev | 0.9672 | 0.3716 | 0.9478 | 5.07 |
| Retriever, bi encoder, top 3 | dev | 0.885 | 0.5752 | 0.8142 | 3.0 |
| Retriever, bi encoder, top 4 | dev | 0.9398 | 0.4777 | 0.899 | 3.84 |
| Retriever, bi encoder, top 5 | dev | 0.9662 | 0.4116 | 0.9407 | 4.58 |
| Retriever, bi encoder + cross encoder rerank, top 2 | dev | 0.8091 | 0.7888 | 0.6897 | 2.0 |
| Retriever, bi encoder + cross encoder rerank, top 3 | dev | 0.9214 | 0.5989 | 0.8735 | 3.0 |
| Retriever, bi encoder + cross encoder rerank, top 4 | dev | 0.9616 | 0.4888 | 0.9394 | 3.84 |
| Retriever, bi encoder + cross encoder rerank, top 5 | dev | 0.9806 | 0.4177 | 0.9694 | 4.58 |

On dev the retriever second pass recovers **-15.7%** of the oracle's gain over the SFT model (paired bootstrap 95% interval [-55.9, 9.8], 2,000 resamples of the 1,534 questions). The trained selector recovers 17.1% [-27.7, 44.6].


### Answer or abstain: learned model against the vote share threshold (v2 part 2)

Both rank the same answers (the 1.5B LoRA r16 vote of 8). LightGBM on 13 features (vote_share, n_results, greedy_error, vote_is_greedy, ret_top, ret_margin, ret_kth, n_tables, n_joins, question_words, evidence_words, db_tables, db_log_mb), fit on val (493 questions); 5 fold CV Brier chose {'num_leaves': 4, 'n_estimators': 50, 'learning_rate': 0.05} with Platt calibration. Brier and ECE (10 bins) score the confidence as a probability that the answer is right; vote share is used as is.

| Split | Score | Brier | ECE | AUROC | Answered at val 70% threshold | EX there |
|---|---|---|---|---|---|---|
| dev | learned (calibrated) | 0.2231 | 0.094 | 0.7223 | 78.41% | 56.57 |
| dev | vote share | 0.229 | 0.1367 | 0.7411 | 81.41% | 55.29 |
| heldback | learned (calibrated) | 0.2765 | 0.3662 | 0.8355 | 82.0% | 84.15 |
| heldback | vote share | 0.1364 | 0.0663 | 0.8495 | 89.0% | 82.02 |

At exactly 70% of dev answered (1073 questions, ties broken by a seeded shuffle): learned **59.18** against vote share **59.46** (only learned right 28, only vote share right 31, exact sign test p = 0.795; accuracy difference 95% bootstrap interval [-1.3, 0.93] points).

Scored against the 1.5B agent's answers instead (what the API returns), on 1533 dev questions: Brier 0.2117, ECE 0.0625, AUROC 0.7346.

### Agent loop against single pass (v2 part 3)

The agent retrieves tables, writes SQL, runs it, and on an error or an empty result inspects the schema or samples column values and rewrites (at most 2 rewrites, 8 tool calls). Options chosen on val. p: exact sign test on the questions exactly one system gets right.

| Model | Split | Agent EX | Single pass EX | p | Retrieved single pass EX | p | Mean steps | Mean tokens | Share repaired |
|---|---|---|---|---|---|---|---|---|---|
| 1.5B SFT LoRA | val | 37.0 | 33.6 | 0.04 | 35.6 | 0.092 | 3.85 | 2907.2 | 0.336 |
| 1.5B SFT LoRA | dev | 44.59 | 43.94 | 0.363 | 43.22 | <0.001 | 3.56 | 1967.6 | 0.2093 |
| 1.5B SFT LoRA | heldback | 68.0 | 68.0 | 1.0 | 68.0 | 1.0 | 3.28 | 1593.9 | 0.11 |
| 1.5B SFT LoRA | iowa | 50.0 | 46.67 | 1.0 | 43.33 | 0.5 | 3.67 | 1915.0 | 0.3 |

1.5B SFT LoRA, 50 dev questions one at a time: agent p50 789.0 ms (p90 3710.9), single pass p50 650.9 ms (p90 987.1).

| 7B SFT DoRA (T4, fp16) | dev | 53.85 | 51.63 | <0.001 | 51.96 | <0.001 | 3.38 | 1585.3 | 0.1375 |
| 7B SFT DoRA (T4, fp16) | heldback | 86.0 | 83.0 | 0.549 | 84.0 | 0.5 | 3.15 | 1342.7 | 0.06 |

The 7B single pass in fp16 on 2x T4 scores 51.63 on dev against 51.69 for the same model in bf16 on the A100 (v1): 9 of 1534 questions differ (p = 1.0).


7B SFT DoRA (T4, fp16), 20 dev questions one at a time: agent p50 2224.9 ms (p90 4104.3), single pass p50 1910.9 ms (p90 3588.9).


### Serving: bf16 against 4 bit AWQ, 1.5B on the RTX 4070 Laptop (v2 part 4)

Dev EX: AWQ 39.7 against bf16 (LoRA adapter, laptop) 43.94 (only AWQ right 51, only bf16 right 116, exact sign test p < 0.001, 1534 questions).

Locust, 60 s per level, dev prompts, greedy, 256 max new tokens. Cost per 1,000 queries assumes this throughput at public on demand prices (L4 $0.8/h, A100_80GB $2.5/h).

| Variant | Concurrency | Requests/s | Completion tokens/s | p50 ms | p99 ms | Peak GPU MiB | Model GiB | KV cache tokens | $ per 1k queries (L4) |
|---|---|---|---|---|---|---|---|---|---|
| bf16_lora | 1 | 1.017 | 63.3 | 960.0 | 2000.0 | 7603 | 3.12 | 94576 | 0.2185 |
| bf16_lora | 8 | 8.233 | 450.7 | 910.0 | 2100.0 | 7603 | 3.12 | 94576 | 0.027 |
| bf16_lora | 32 | 21.167 | 1115.0 | 1400.0 | 3500.0 | 7603 | 3.12 | 94576 | 0.0105 |
| bf16_merged | 1 | 1.217 | 75.4 | 790.0 | 1700.0 | 7043 | 2.98 | 102736 | 0.1826 |
| bf16_merged | 8 | 10.2 | 537.4 | 720.0 | 1600.0 | 7043 | 2.98 | 102736 | 0.0218 |
| bf16_merged | 32 | 25.65 | 1360.9 | 1100.0 | 2800.0 | 7043 | 2.98 | 102736 | 0.0087 |
| awq_w4a16 | 1 | 2.7 | 171.7 | 360.0 | 620.0 | 7397 | 1.12 | 184592 | 0.0823 |
| awq_w4a16 | 8 | 19.55 | 1023.7 | 380.0 | 1000.0 | 7397 | 1.12 | 184592 | 0.0114 |
| awq_w4a16 | 32 | 34.667 | 1843.1 | 850.0 | 2200.0 | 7397 | 1.12 | 184592 | 0.0064 |

### Serving: bf16 against 4 bit AWQ, 7B DoRA, 4 bit on the RTX 4070 Laptop (bf16 7B does not fit 8 GB) (v2 part 4)

Dev EX: AWQ 51.5 against bf16 (merged, A100, v1) 51.69 (only AWQ right 46, only bf16 right 49, exact sign test p = 0.838, 1534 questions).

Locust, 60 s per level, dev prompts, greedy, 256 max new tokens. Cost per 1,000 queries assumes this throughput at public on demand prices (L4 $0.8/h, A100_80GB $2.5/h).

| Variant | Concurrency | Requests/s | Completion tokens/s | p50 ms | p99 ms | Peak GPU MiB | Model GiB | KV cache tokens | $ per 1k queries (L4) |
|---|---|---|---|---|---|---|---|---|---|
| awq_w4a16_7b | 1 | 0.783 | 50.6 | 1200.0 | 4200.0 | 7931 | 5.29 | 13552 | 0.2838 |
| awq_w4a16_7b | 8 | 6.183 | 325.8 | 1200.0 | 2900.0 | 7931 | 5.29 | 13552 | 0.0359 |
| awq_w4a16_7b | 32 | 10.75 | 511.6 | 2700.0 | 6700.0 | 7931 | 5.29 | 13552 | 0.0207 |

### Live demo warehouse: Iowa Liquor Sales (v2 part 6)

30 questions written for this project with gold SQL validated by execution (`data/warehouse/iowa_questions.json`, **needs human review**), over 2,000,000 invoice lines (2022-01-02 to 2022-10-19). A second held back set: scored only. 95% Wilson interval; p is an exact sign test against the single pass row.

| System | EX [95% CI] | Only this right | Only single pass right | p |
|---|---|---|---|---|
| 1.5B SFT LoRA, single pass | 46.67 [30.23, 63.86] | 0 | 0 | 1.0 |
| 1.5B SFT LoRA, retrieved tables | 43.33 [27.38, 60.8] | 0 | 1 | 1.0 |
| 1.5B agent | 50.0 [33.15, 66.85] | 2 | 1 | 1.0 |

### Targets (not yet measured)

These runs are configured but have no logged results yet. No numbers are reported for them.

* 7B SFT DoRA, oracle tables (upper bound) (`configs/linking/53_qwen7b_dora_oracle.yaml`, A100 80GB)
* 7B SFT DoRA, self linked tables (2 pass) (`configs/linking/54_qwen7b_dora_selflink.yaml`, A100 80GB)

![dev_ex_by_difficulty.png](docs/figures/dev_ex_by_difficulty.png)
![learn_vs_forget.png](docs/figures/learn_vs_forget.png)
![accuracy_vs_coverage.png](docs/figures/accuracy_vs_coverage.png)
![grpo_rewards.png](docs/figures/grpo_rewards.png)