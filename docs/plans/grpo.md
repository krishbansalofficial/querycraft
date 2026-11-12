# Plan: reinforcement learning with an execution reward (GRPO)

Status: planned, not started. Nothing here runs on Modal until the 7B study is finished and the 1.5B result
justifies it.

## Question

Does reinforcement learning with a reward computed by running the query improve on supervised fine tuning,
and does shaping that reward with the project's error categories help compared with a plain right or wrong
reward?

## Arms (1.5B, laptop)

| Arm | Start | Reward |
|---|---|---|
| A | SFT LoRA r16, selected checkpoint (exists) | none (baseline) |
| B | A | binary: 1 if the result rows match gold, else 0 |
| C | A | shaped by error category (below) |

B and C share prompts, order, seed, steps, group size and learning rate. Only the reward differs.

## Shaped reward

| Outcome | Reward |
|---|---|
| Correct result | 1.0 |
| Runs, wrong result | 0.1, plus 0.2 if the tables match gold, 0.2 if the joins match, 0.1 if aggregation matches, 0.1 if the columns match (at most 0.7) |
| Does not run | 0.0, or minus 0.2 for a syntax error |

A wrong query can never earn as much as a correct one. The risk is reward hacking: the model learns the
right tables without the right answer. We track the gap between mean shaped reward and execution accuracy on
val; a rising gap with flat accuracy means hacking, and arm B is the control for exactly that.

## Data

1. Sample 8 answers per train question with the SFT model (reuses `querycraft.confidence`).
2. Keep questions it solves sometimes but not always (1 to 7 of 8 correct). All right or all wrong groups
   give GRPO no learning signal.
3. Cap at 2,000 prompts. Train databases only; val picks the checkpoint; dev and held back only report.

## Recipe (TRL `GRPOTrainer`)

* Group size 8, temperature 1.0, max completion 256 tokens
* LoRA r16 on top of the SFT adapter, lr 1e-5, no KL penalty (beta 0), 250 steps at 8 prompts per step
* Checkpoints every 50 steps; the best on val goes to dev
* Generation through vLLM in colocate mode if it fits 8 GB, otherwise Hugging Face generate

## Evaluation

The existing harness on val, dev and held back, the forgetting probe, and the confidence pipeline.
RL often makes a model more certain, so we also check whether the confidence AUROC improves or degrades.

## Steps and time

| Step | Time |
|---|---|
| 1. Reward function with unit tests (each error category maps to the table above) | 0.5 day |
| 2. Learnable prompt filter on train (about 1 h of laptop GPU) | 0.5 day |
| 3. Feasibility probe: 10 GRPO steps, measure seconds per step and peak memory | 0.5 day |
| 4. Arms B and C on the laptop (estimate 3 to 4 h each, firmed up by step 3) | 1 day |
| 5. Evaluate, add a results table and the reward hacking plot | 0.5 day |
| 6. Only if 1.5B gains: 7B arms on Modal in a later billing cycle, under the same spend cap | later |

## Guardrails

* The reward only ever runs on train databases (the leakage test covers this).
* Fixed budget: 250 steps per arm, decided now, not after seeing dev.
* If the probe shows more than 6 h per arm on the laptop, cut to 1,000 prompts rather than renting.
