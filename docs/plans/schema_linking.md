# Plan: schema linking (pick the tables before writing the SQL)

Status: step 1 passed the gate; self linking (no training) evaluated; trained selector (step 3) next.

## Step 1 result: oracle upper bound

Showing the SFT LoRA r16 model only its gold SQL's tables (same pruner, same 1,280 token budget) lifts dev EX from
43.94 to 48.50 (134 newly right, 64 newly wrong, p < 0.001) and held back from 68 to 79 (p = 0.013). Val: 33.6 to
45.0. `wrong_table` errors on dev halve (241 to 116). The gate (3 points, p < 0.05) is passed.

## Step 3, cheaper version first: self linking

Before training a separate selector: the SFT model's own greedy prediction (already on disk) picks the tables, plus
one foreign key hop, and a second pass writes the SQL over only those. On val this keeps every gold table for 96% of
questions while showing 9.3 of 42 tables on average (dev: 97.4%, 5.1 of 7.1). The hop rule was fixed before looking
at any accuracy.

Result: val 33.6 to 36.6, dev 43.94 to 44.46 (39 newly right, 31 newly wrong, p = 0.40), held back 68 to 76
(8 newly right, 0 newly wrong, p = 0.008). Wrong table errors on dev do not fall (241 to 244): the first pass's own
mistakes stay in the schema. Dev databases average 7 tables, so removing about 2 changes little; the oracle gains by
showing about 2 tables. A selector needs much higher precision than the first pass, which is what step 3 trains for.

## Why

Schema errors dominate: for the 1.5B LoRA r16 model, `wrong_table` (241) and `wrong_column` (268) are 509 of its
860 wrong dev answers. The existing pruner already keeps the gold tables in the prompt for 99.35% of dev questions
(`data/processed/prompt_stats.json`), so these errors come from the model choosing among tables it can see,
not from missing tables. The question is whether a smaller, more focused schema helps it choose.

## Steps

| Step | What | Cost |
|---|---|---|
| 1. Oracle upper bound | The SFT LoRA r16 model, shown only the tables its gold SQL uses (`configs/linking/50_*`) | about 30 min eval |
| 2. Gate | Continue only if the oracle beats the SFT model by at least 3 dev points (paired test p < 0.05). Otherwise stop and report the oracle as the ceiling | none |
| 3. Table selector | The same 1.5B model, LoRA fine tuned on train to list the tables a question needs (targets from the gold SQL); recall and precision on val | about 2 h |
| 4. Two pass system | Selector picks tables (plus one foreign key hop for safety), SQL model writes the query over only those tables; SQL model retrained on that prompt format | about 3 h |
| 5. Report | Paired test against the SFT model, selector recall, error categories | 0.5 h |

## Guardrails

* The oracle uses gold labels at inference. It is reported only as an upper bound, labelled as such, never as a result.
* Every choice (threshold, which hop rule, which checkpoint) is made on val; dev and held back are only evaluated.
* Laptop only. A 7B version waits for the Modal credit cycle and the user's go ahead.
