"""Score predictions by execution. Pure CPU; consumes predictions jsonl written by infer.py."""
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

from .execution import ERROR_TAGS, score_many


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, round(100 * (c - h), 2)), round(100 * (c + h), 2))  # max() drops the -0.0 float artefact


def summarize(preds, verdicts):
    n = len(preds)
    ex = sum(v["ex"] for v in verdicts)
    m = {"n": n, "ex": round(100 * ex / max(n, 1), 2), "ex_ci95": wilson(ex, n),
         "ex_bird": round(100 * sum(v["ex_bird"] for v in verdicts) / max(n, 1), 2),
         "gold_errors": sum(v["gold_error"] is not None for v in verdicts),
         "errors": {t: 0 for t in ERROR_TAGS}}
    m["errors"].update(Counter(v["error_tag"] for v in verdicts if v["error_tag"]))
    by = {}
    for p, v in zip(preds, verdicts):
        if p.get("difficulty"):
            by.setdefault(p["difficulty"], []).append(v["ex"])
    m["by_difficulty"] = {k: {"n": len(x), "ex": round(100 * sum(x) / len(x), 2)} for k, x in sorted(by.items())}
    return m


def score_file(pred_path, workers=None):
    pred_path = Path(pred_path)
    preds = [json.loads(l) for l in pred_path.read_text().splitlines()]
    verdicts = score_many([{"db_path": p["db_path"], "pred_sql": p["pred_sql"], "gold_sql": p["gold_sql"]} for p in preds],
                          workers=workers or min(16, os.cpu_count() or 4))
    with open(pred_path.with_suffix(".scored.jsonl"), "w") as f:
        for p, v in zip(preds, verdicts):
            f.write(json.dumps({**{k: p[k] for k in ("id", "db_id", "difficulty", "pred_sql", "gold_sql")}, **v}) + "\n")
    m = summarize(preds, verdicts)
    pred_path.with_suffix(".metrics.json").write_text(json.dumps(m, indent=2))
    return m


if __name__ == "__main__":
    print(json.dumps(score_file(sys.argv[1]), indent=2))
