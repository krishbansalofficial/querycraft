# Held back set review

The 100 held back questions (`data/heldback/heldback.json`) were drafted by an AI assistant. This file records a
second review pass, also done by an AI assistant, on 2026-11-14. **No human has reviewed the set yet.**

## What was checked, for every question

1. The gold query runs on the database and returns a non empty result (`python data/heldback/draft.py`).
2. The gold query answers the question as written, read by hand against the schema and the evidence.
3. For `ORDER BY ... LIMIT k` gold queries: whether the k-th and (k+1)-th rows tie on the sort key. A tie means more
   than one answer is correct but only one is accepted.
4. NULLs in the gold result, duplicate rows without `DISTINCT`, and duplicate entity names (Batman, Superman,
   Wolverine and Godzilla are each unique, so name filters are safe).

## Findings

| Question | Issue | Effect |
|---|---|---|
| student_club_04 "Which member attended the most events?" | Elijah Allen and Phillip Cullen tie at 16 events; gold returns one | a correct answer naming the other member is scored wrong |
| student_club_24 "Which event had the most attendees?" | Registration and Yearly Kickoff tie at 30; gold returns one | same |
| superhero_24 "five tallest female superheroes, tallest first" | Bloodaxe and Thundra tie at 218 cm, so their order is arbitrary | none observed: every model was scored correct |
| superhero_30 "Which gender has the highest average weight?" | answer is 'N/A' because of implausible weights in the source data | the gold is right for the data as stored; kept |
| superhero_03 average female height | includes Giganta at 6,250 cm (source data) | the gold is right for the data as stored; kept |

The other 95 questions had no issue found.

## Decision

The gold answers were **not** changed. Editing gold after seeing every model's answers would let the results shape
the test. Instead, held back EX was recomputed without the two tied questions:

| Model | Held back EX (100) | Without the 2 tied questions (98) |
|---|---|---|
| qwen7b_fewshot3 | 90 | 89.8 |
| qwen7b_zeroshot | 88 | 87.8 |
| qwen7b_qlora_r16 | 84 | 84.7 |
| qwen7b_dora_best | 83 | 83.7 |
| qwen7b_qlora_r64 | 83 | 83.7 |
| qwen7b_qlora_r8 | 82 | 82.7 |
| qwen1.5b_lora_r16_oracle | 79 | 79.6 |
| qwen1.5b_lora_r16_selflink | 76 | 76.5 |
| qwen1.5b_grpo_binary | 73 | 74.5 |
| qwen1.5b_grpo_shaped | 73 | 73.5 |
| qwen1.5b_lora_r16 | 68 | 68.4 |
| qwen1.5b_qlora_r16 | 66 | 66.3 |
| qwen1.5b_full | 65 | 66.3 |
| qwen1.5b_lora_r16_selector | 63 | 63.3 |
| qwen1.5b_zeroshot | 61 | 61.2 |
| qwen1.5b_fewshot3 | 49 | 49.0 |

The largest change is 1.5 points. No model moves past another: the only differences are that 1.5B full fine
tuning now ties QLoRA r16 (66.3 each) and the two GRPO arms no longer tie (74.5 and 73.5). No conclusion in the
README depends on the two tied questions.

## Still open

A human review of all 100 questions. Until then the set stays labelled "AI drafted, not yet human reviewed".
