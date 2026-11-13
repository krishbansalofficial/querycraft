"""GRPO rewards computed by running the query (docs/plans/grpo.md).

binary:  1 if the result matches gold, else 0.
shaped:  1 if correct; a query that runs but is wrong earns 0.1 plus partial credit for matching gold's
         tables (0.2), joins (0.2), aggregation (0.1) and columns (0.1), at most 0.7; a query that fails
         earns 0, or -0.2 for a syntax error. A wrong query can never reach a correct one's reward.

Both follow TRL's reward function signature: f(prompts, completions, db_path, gold_sql, **kw) -> list[float].
"""
from functools import lru_cache

from .execution import _structure, execute, has_top_level_order_by, results_match, tag_error
from .prompts import extract_sql


@lru_cache(maxsize=4096)  # each prompt's gold runs once, not once per sample in the group
def _gold(db_path, gold_sql):
    rows, err = execute(db_path, gold_sql, 60.0)
    return rows, err, has_top_level_order_by(gold_sql)


def _text(c):
    return c if isinstance(c, str) else c[-1]["content"]  # plain or conversational completions


def outcome(db_path, gold_sql, pred_sql):
    """('correct' | 'wrong' | 'failed', pred_error)."""
    gold_rows, gold_err, ordered = _gold(db_path, gold_sql)
    if gold_err is not None:
        raise ValueError(f"gold query fails on {db_path}: {gold_err}")  # such prompts are filtered out upstream
    rows, err = execute(db_path, pred_sql, 30.0) if pred_sql.strip() else (None, "empty prediction")
    if err is not None:
        return "failed", err
    return ("correct" if results_match(rows, gold_rows, ordered) else "wrong"), None


def binary_one(db_path, gold_sql, pred_sql):
    return 1.0 if outcome(db_path, gold_sql, pred_sql)[0] == "correct" else 0.0


def shaped_one(db_path, gold_sql, pred_sql):
    kind, err = outcome(db_path, gold_sql, pred_sql)
    if kind == "correct":
        return 1.0
    if kind == "failed":
        return -0.2 if tag_error(pred_sql, gold_sql, err) == "syntax" else 0.0
    p, g = _structure(pred_sql), _structure(gold_sql)
    if p is None or g is None:
        return 0.1
    return round(0.1 + 0.2 * (p["tables"] == g["tables"]) + 0.2 * (p["joins"] == g["joins"])
                 + 0.1 * (p["aggs"] == g["aggs"] and p["group"] == g["group"])
                 + 0.1 * (p["columns"] == g["columns"]), 4)


def binary_reward(prompts, completions, db_path, gold_sql, **kw):
    return [binary_one(d, g, extract_sql(_text(c))) for c, d, g in zip(completions, db_path, gold_sql)]


def shaped_reward(prompts, completions, db_path, gold_sql, **kw):
    return [shaped_one(d, g, extract_sql(_text(c))) for c, d, g in zip(completions, db_path, gold_sql)]
