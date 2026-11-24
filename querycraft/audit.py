"""Audit: every reported number must be recomputable from raw per example files.

1. Recount EX from each *.scored.jsonl and compare with results.json (the source of README numbers).
2. Re-execute a random sample of predictions with an independent checker that shares no code with
   querycraft.execution, and report agreement with the harness verdicts.
3. Recount forgetting pass@1 and confidence vote accuracy from their raw files.

    python -m querycraft.audit [runs_dir] [--sample 150]
"""
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def _local(db_path):
    """Runs made on Modal record /vol/data/... paths; map them onto this machine's data folder."""
    return db_path if Path(db_path).exists() else db_path.replace("/vol/data", "data", 1)


def independent_ex(db_path, pred, gold, timeout_steps=20_000_000):
    """Minimal checker: run both queries read only; compare row multisets, or lists when gold ends with ORDER BY."""
    db_path = _local(db_path)
    def run(sql):
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode(errors="replace")
        n = [0]
        def tick():
            n[0] += 1
            return n[0] > timeout_steps // 1000
        con.set_progress_handler(tick, 1000)
        try:
            return con.execute(sql).fetchall()
        finally:
            con.close()
    try:
        g = run(gold)
    except Exception:
        return None  # broken gold: not judgeable
    try:
        p = run(pred)
    except Exception:
        return False
    ordered = re.search(r"order\s+by[^()]*$", gold, re.I | re.S) is not None
    return p == g if ordered else Counter(p) == Counter(g)


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()]


def main(runs_dir="runs", sample=150):
    rows, problems = [], []
    rng = random.Random(0)
    for res_f in sorted(Path(runs_dir).glob("*/results.json")):
        run = res_f.parent
        res = json.loads(res_f.read_text())
        tag = res.get("selected_checkpoint", "base")
        for split in ("val", "dev", "heldback", "iowa"):
            if "ex" not in res.get(split, {}):  # missing split, or a table selector run (set metrics, no SQL)
                continue
            agent = "options" in res  # v2 agent runs keep per question files next to results.json
            scored = run / (f"{split}_agent.scored.jsonl" if agent else f"eval/{split}_{tag}.scored.jsonl")
            preds = run / (f"{split}_agent.jsonl" if agent else f"eval/{split}_{tag}.jsonl")
            if not scored.exists():
                problems.append(f"{run.name} {split}: no scored file")
                continue
            s, p = _jsonl(scored), _jsonl(preds)
            recount = round(100 * sum(x["ex"] for x in s) / len(s), 2)
            ok = recount == res[split]["ex"]
            pick = rng.sample(range(len(s)), min(sample, len(s)))
            agree = judged = 0
            for i in pick:
                v = independent_ex(p[i]["db_path"], p[i]["pred_sql"], p[i]["gold_sql"])
                if v is None:
                    continue
                judged += 1
                agree += v == s[i]["ex"]
            rows.append((run.name, split, res[split]["ex"], recount, len(s), f"{agree}/{judged}"))
            if not ok:
                problems.append(f"{run.name} {split}: results.json {res[split]['ex']} != recount {recount}")
        if "val_ex_by_checkpoint" in res:  # selection must come from val files, never dev
            for ck, v in res["val_ex_by_checkpoint"].items():
                s = _jsonl(run / "eval" / f"val_{ck}.scored.jsonl")
                if round(100 * sum(x["ex"] for x in s) / len(s), 2) != v:
                    problems.append(f"{run.name} val {ck}: logged {v} does not match its file")
        f = run / "eval" / f"forgetting_{tag}.jsonl"
        if f.exists() and "forgetting" in res:
            fr = _jsonl(f)
            he = [x["passed"] for x in fr if x["id"].startswith("HumanEval")]
            if round(100 * sum(he) / len(he), 2) != res["forgetting"]["humaneval"]["pass@1"]:
                problems.append(f"{run.name}: HumanEval recount mismatch")
        cf = run / "confidence" / "confidence.json"
        if cf.exists():
            c = json.loads(cf.read_text())
            recs = [r for r in _jsonl(run / "confidence" / f"dev_T{c['temperature']}.jsonl") if not r["gold_error"]]
            if round(100 * sum(r["ex"] for r in recs) / len(recs), 2) != c["dev"]["vote_ex"]:
                problems.append(f"{run.name}: confidence vote_ex recount mismatch")
    print(f"{'run':24} {'split':9} {'reported':>8} {'recount':>8} {'n':>5}  independent re-execution agrees")
    for r in rows:
        print(f"{r[0]:24} {r[1]:9} {r[2]:8} {r[3]:8} {r[4]:5}  {r[5]}")
    print("\nPROBLEMS:" if problems else "\nNo mismatches: every reported number matches its raw files.")
    for p in problems:
        print(" ", p)
    return problems


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0] if a else "runs", int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 150)
