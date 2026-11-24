"""Drift report (v2 part 5): are live questions still like the ones the system was tuned on?

Reads the API's question log (runs/live/questions.sqlite, one row per /ask) and compares it with val, the split every
choice was made on:
  * question length (words): two sample Kolmogorov Smirnov test against val questions
  * confidence: two sample KS test against the abstain model's out of fold val predictions
  * database: share of live questions on databases the system never saw in train, val or dev, plus a chi square
    test of the database mix in the newest window against the window before it
A shift is flagged at p < ALPHA. Small samples are reported but never flagged (MIN_N).

    python -m querycraft.drift [--log runs/live/questions.sqlite] [--window-days 7]
"""
import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

ALPHA, MIN_N = 0.01, 30


def ks(a, b):
    from scipy.stats import ks_2samp
    r = ks_2samp(a, b)
    return {"statistic": round(float(r.statistic), 4), "p": float(r.pvalue)}


def chi_mix(cur, prev):
    """Chi square test of the database mix in two windows (contingency table)."""
    from scipy.stats import chi2_contingency
    keys = sorted(set(cur) | set(prev))
    if len(keys) < 2:
        return None
    r = chi2_contingency([[cur.get(k, 0) for k in keys], [prev.get(k, 0) for k in keys]])
    return {"statistic": round(float(r.statistic), 3), "p": float(r.pvalue), "dof": int(r.dof)}


def reference():
    from .data import load_split
    val = load_split("val")
    known = {r["db_id"] for s in ("train", "val", "dev") for r in load_split(s)}
    ab = Path("runs/abstain_lgbm/results.json")
    conf = json.loads(ab.read_text())["fit"].get("val_oof") if ab.exists() else None
    return {"question_words": [len(r["question"].split()) for r in val], "confidence": conf, "known_dbs": known}


def report(log, window_days=7, ref=None, now=None):
    ref = ref or reference()
    now = now or time.time()
    with sqlite3.connect(log) as c:
        rows = c.execute("SELECT ts, db, question, confidence, abstained, outcome FROM questions").fetchall()
    cur = [r for r in rows if r[0] >= now - window_days * 86400]
    prev = [r for r in rows if now - 2 * window_days * 86400 <= r[0] < now - window_days * 86400]
    out = {"n_window": len(cur), "n_previous_window": len(prev), "window_days": window_days, "alpha": ALPHA, "checks": {}}
    if not cur:
        return out
    qw = [len(r[2].split()) for r in cur]
    out["checks"]["question_words"] = {"live_mean": round(sum(qw) / len(qw), 2),
                                       "val_mean": round(sum(ref["question_words"]) / len(ref["question_words"]), 2),
                                       **ks(qw, ref["question_words"])}
    cf = [r[3] for r in cur if r[3] is not None]
    if cf and ref.get("confidence"):
        out["checks"]["confidence"] = {"live_mean": round(sum(cf) / len(cf), 4),
                                       "val_mean": round(sum(ref["confidence"]) / len(ref["confidence"]), 4),
                                       **ks(cf, ref["confidence"])}
    new = sum(r[1] not in ref["known_dbs"] for r in cur) / len(cur)
    mix = chi_mix(Counter(r[1] for r in cur), Counter(r[1] for r in prev)) if len(prev) >= MIN_N else None
    out["checks"]["database"] = {"share_on_unseen_databases": round(new, 4),
                                 "top": Counter(r[1] for r in cur).most_common(5), "mix_vs_previous_window": mix}
    out["outcomes"] = dict(Counter(r[5] for r in cur))
    flags = []
    if len(cur) >= MIN_N:
        for k in ("question_words", "confidence"):
            if k in out["checks"] and out["checks"][k]["p"] < ALPHA:
                flags.append(k)
        if mix and mix["p"] < ALPHA:
            flags.append("database_mix")
    out["flags"] = flags
    return out


def markdown(r):
    lines = [f"# Drift report ({time.strftime('%Y-%m-%d %H:%M')})", "",
             f"{r['n_window']} questions in the last {r['window_days']} days ({r['n_previous_window']} in the window "
             f"before). Flagged at p < {r['alpha']} when at least {MIN_N} questions are logged.", ""]
    for k, v in r.get("checks", {}).items():
        lines.append(f"* **{k}**: " + ", ".join(f"{a} = {b}" for a, b in v.items()))
    lines += ["", f"Flags: {', '.join(r.get('flags', [])) or 'none'}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/live/questions.sqlite")
    ap.add_argument("--window-days", type=float, default=7)
    a = ap.parse_args()
    r = report(a.log, a.window_days)
    Path(a.log).with_name("drift_report.json").write_text(json.dumps(r, indent=2))
    Path(a.log).with_name("drift_report.md").write_text(markdown(r))
    print(markdown(r))


if __name__ == "__main__":
    main()
