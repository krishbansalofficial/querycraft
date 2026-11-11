"""Execution harness: run SQL on SQLite with a timeout, compare results, tag errors.

Two match criteria are reported:
  ex       order insensitive (multiset) unless the gold query has a top level ORDER BY,
           in which case row order must match. This is the primary metric.
  ex_bird  the official BIRD criterion, set(pred) == set(gold), for leaderboard comparability.
"""
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import sqlglot
from sqlglot import exp

ERROR_TAGS = ["syntax", "wrong_table", "wrong_column", "wrong_join", "wrong_aggregation", "timeout", "other"]


class QueryTimeout(Exception):
    pass


def execute(db_path, sql, timeout=30.0, max_rows=100_000):
    """Return (rows, error_string|None). Read only connection, wall clock timeout."""
    deadline = time.monotonic() + timeout
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode(errors="replace")
    # returning nonzero from the handler aborts the statement
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        rows = conn.execute(sql).fetchmany(max_rows)
        return rows, None
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            return None, "timeout"
        return None, str(e)
    except Exception as e:  # sqlite3.Warning (multiple statements), ProgrammingError, ...
        return None, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def _parse(sql):
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def has_top_level_order_by(sql) -> bool:
    tree = _parse(sql)
    if tree is None:
        return "order by" in sql.lower()
    return tree.args.get("order") is not None


def results_match(pred_rows, gold_rows, ordered: bool) -> bool:
    if ordered:
        return list(pred_rows) == list(gold_rows)
    return Counter(pred_rows) == Counter(gold_rows)


def _structure(sql):
    tree = _parse(sql)
    if tree is None:
        return None
    return {
        "tables": {t.name.lower() for t in tree.find_all(exp.Table)},
        "columns": {c.name.lower() for c in tree.find_all(exp.Column)},
        # join signature: joined table + the column names in its ON clause (aliases ignored)
        "joins": sorted(
            ((j.this.name.lower() if isinstance(j.this, exp.Table) else "sub"),
             tuple(sorted(c.name.lower() for c in (j.args.get("on") or exp.Null()).find_all(exp.Column))))
            for j in tree.find_all(exp.Join)
        ),
        "aggs": sorted(type(a).__name__ for a in tree.find_all(exp.AggFunc)),
        "group": sorted(g.sql().lower() for grp in tree.find_all(exp.Group) for g in grp.expressions),
    }


def tag_error(pred_sql, gold_sql, pred_error):
    """Assign one error category to a wrong prediction. Execution errors come first,
    then a structural diff against gold (tables > joins > aggregation > columns)."""
    if pred_error == "timeout":
        return "timeout"
    if pred_error is not None:
        e = pred_error.lower()
        if "no such table" in e:
            return "wrong_table"
        if "no such column" in e or "ambiguous column" in e:
            return "wrong_column"
        if "syntax" in e or "incomplete input" in e or "unrecognized token" in e or not pred_sql.strip():
            return "syntax"
        return "other"
    p, g = _structure(pred_sql), _structure(gold_sql)
    if p is None:
        return "syntax"
    if g is None:
        return "other"
    if p["tables"] != g["tables"]:
        return "wrong_table"
    if p["joins"] != g["joins"]:
        return "wrong_join"
    if p["aggs"] != g["aggs"] or p["group"] != g["group"]:
        return "wrong_aggregation"
    if p["columns"] != g["columns"]:
        return "wrong_column"
    return "other"  # same skeleton, wrong literal / filter / ordering


def score_one(item, timeout=30.0, gold_timeout=60.0):
    """item: dict(db_path, pred_sql, gold_sql). Returns per item verdict dict."""
    t0 = time.monotonic()
    gold_rows, gold_err = execute(item["db_path"], item["gold_sql"], gold_timeout)
    pred_rows, pred_err = execute(item["db_path"], item["pred_sql"], timeout) if item["pred_sql"].strip() else (None, "empty prediction")
    out = {"gold_error": gold_err, "pred_error": pred_err, "ex": False, "ex_bird": False, "error_tag": None}
    if gold_err is None and pred_err is None:
        out["ex"] = results_match(pred_rows, gold_rows, has_top_level_order_by(item["gold_sql"]))
        out["ex_bird"] = set(pred_rows) == set(gold_rows)
    if not out["ex"]:
        out["error_tag"] = "other" if gold_err else tag_error(item["pred_sql"], item["gold_sql"], pred_err)
    out["exec_s"] = round(time.monotonic() - t0, 3)
    return out


def score_many(items, workers=8, timeout=30.0):
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(score_one, items, [timeout] * len(items), chunksize=4))
