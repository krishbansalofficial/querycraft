"""Knowing when it is wrong: execution guided self consistency with abstention.

For each question, sample N queries, run them, and group them by the rows they return. The largest group's
query is the answer; the group's share of the N samples is the confidence (failed queries count as no vote).
The model answers only when confidence >= a threshold, trading coverage for accuracy.

Every choice (sampling temperature, thresholds) is made on val; dev and held back only report.

    python -m querycraft.confidence configs/local/11_qwen1.5b_lora_r16.yaml
"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from .config import load_config, logged_stage, run_dir
from .execution import execute, score_one

TARGETS = (0.6, 0.7, 0.8)   # selective accuracy targets used to pick thresholds on val
VOTE_TIMEOUT = 15.0         # ponytail: half the scoring timeout; slow samples just lose their vote


def signature(rows):
    """Order insensitive multiset of rows; repr() makes mixed types hashable and comparable."""
    return tuple(sorted(Counter(map(repr, rows)).items()))


def vote(item):
    """item: db_path, samples [sql], greedy sql. Returns (chosen sql, confidence, n distinct results)."""
    n = len(item["samples"])
    by_sql = {}
    for s in set(item["samples"]):
        rows, err = execute(item["db_path"], s, VOTE_TIMEOUT) if s.strip() else (None, "empty")
        by_sql[s] = None if err else signature(rows)
    groups = Counter(by_sql[s] for s in item["samples"] if by_sql[s] is not None)
    if not groups:
        return item["greedy"], 0.0, 0
    g_sig = by_sql.get(item["greedy"])
    top = max(groups.values())
    # ties: prefer the group that contains the greedy answer, else the first sampled
    win = g_sig if groups.get(g_sig) == top else next(by_sql[s] for s in item["samples"] if groups.get(by_sql[s]) == top)
    chosen = item["greedy"] if by_sql.get(item["greedy"]) == win else next(s for s in item["samples"] if by_sql[s] == win)
    return chosen, groups[win] / n, len(groups)


def auroc(conf, correct):
    """P(conf of a correct answer > conf of a wrong one), ties count half (Mann Whitney U)."""
    pos = [c for c, y in zip(conf, correct) if y]
    neg = [c for c, y in zip(conf, correct) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def selective(conf, correct, t):
    """Coverage and accuracy when answering only questions with conf >= t."""
    kept = [y for c, y in zip(conf, correct) if c >= t]
    return {"threshold": t, "coverage": round(100 * len(kept) / len(conf), 2),
            "accuracy": round(100 * sum(kept) / len(kept), 2) if kept else None}


def thresholds_for_targets(conf, correct, targets=TARGETS):
    """Lowest threshold whose selective accuracy reaches each target (max coverage at that accuracy)."""
    out = {}
    for target in targets:
        ok = [t for t in sorted(set(conf)) if (s := selective(conf, correct, t))["accuracy"] is not None
              and s["accuracy"] >= 100 * target]
        out[str(target)] = ok[0] if ok else None
    return out


def greedy_sql(rdir, split, tag, rows):
    """The model's greedy answers from the normal evaluation (tie breaker in the vote)."""
    f = Path(rdir) / "eval" / f"{split}_{tag}.jsonl"
    by_id = {p["id"]: p["pred_sql"] for p in map(json.loads, f.read_text().splitlines())}
    return [by_id[r["id"]] for r in rows]


def _score_votes(rows, samples, greedy):
    items = [{"db_path": r["db_path"], "samples": ss, "greedy": g} for r, ss, g in zip(rows, samples, greedy)]
    with ProcessPoolExecutor(8, mp_context=get_context("spawn")) as pool:  # spawn: the parent holds a CUDA context
        votes = list(pool.map(vote, items, chunksize=4))
        verdicts = list(pool.map(score_one, [{"db_path": r["db_path"], "pred_sql": v[0], "gold_sql": r["sql"]}
                                             for r, v in zip(rows, votes)], chunksize=4))
    return [{"id": r["id"], "difficulty": r["difficulty"], "chosen_sql": v[0], "confidence": v[1], "n_results": v[2],
             "ex": d["ex"], "gold_error": d["gold_error"]} for r, v, d in zip(rows, votes, verdicts)]


def _summary(recs):
    recs = [r for r in recs if not r["gold_error"]]  # broken gold queries cannot be judged
    conf, correct = [r["confidence"] for r in recs], [r["ex"] for r in recs]
    return {"n": len(recs), "vote_ex": round(100 * sum(correct) / len(recs), 2), "auroc": auroc(conf, correct),
            "curve": [selective(conf, correct, t) for t in sorted(set(conf))]}


def main(cfg_path):
    from .data import load_split
    from .infer import Generator
    from .prompts import PromptBuilder, extract_sql
    from .run_eval import _serving

    cfg = load_config(cfg_path)
    c = cfg.get("confidence", {"n": 8, "temperatures": [0.6, 0.8, 1.0]})
    res = json.loads((run_dir(cfg) / "results.json").read_text())
    tag = res.get("selected_checkpoint", "base")
    ckpt = run_dir(cfg) / "ckpt" / tag if tag != "base" else None
    out = run_dir(cfg) / "confidence"
    out.mkdir(exist_ok=True)
    with logged_stage(cfg, "confidence") as meta:
        model_path, adapter = _serving(cfg, ckpt)
        gen = Generator(cfg, model_path)
        pb = PromptBuilder(gen.tok, cfg["prompt"]["schema_budget"])
        summary = {"checkpoint": tag, "n": c["n"], "val_by_temperature": {}}

        def run(split, temp):
            f = out / f"{split}_T{temp}.jsonl"
            if f.exists():
                return [json.loads(l) for l in f.read_text().splitlines()]
            rows = load_split(split)
            if cfg["eval"].get("limit"):
                rows = rows[: cfg["eval"]["limit"]]
            samples = gen.generate([pb.prompt_text(r) for r in rows], adapter, n=c["n"], temperature=temp)
            samples = [[extract_sql(s) for s in ss] for ss in samples]
            recs = _score_votes(rows, samples, greedy_sql(run_dir(cfg), split, tag, rows))
            f.write_text("".join(json.dumps(r) + "\n" for r in recs))
            return recs

        for temp in c["temperatures"]:
            summary["val_by_temperature"][str(temp)] = _summary(run("val", temp))
        # pick the temperature whose confidence best separates right from wrong on val
        best_t = max(c["temperatures"], key=lambda t: (summary["val_by_temperature"][str(t)]["auroc"] or 0,
                                                       summary["val_by_temperature"][str(t)]["vote_ex"]))
        val = [r for r in run("val", best_t) if not r["gold_error"]]
        picks = thresholds_for_targets([r["confidence"] for r in val], [r["ex"] for r in val])
        summary |= {"temperature": best_t, "thresholds_from_val": picks}
        for split in ("dev", "heldback"):
            recs = run(split, best_t)
            s = _summary(recs)
            judged = [r for r in recs if not r["gold_error"]]
            s["at_val_thresholds"] = {k: (selective([r["confidence"] for r in judged], [r["ex"] for r in judged], t)
                                          if t is not None else None) for k, t in picks.items()}
            summary[split] = s
        (out / "confidence.json").write_text(json.dumps(summary, indent=2))
        meta["confidence"] = "confidence/confidence.json"
    print(json.dumps({k: summary[k] for k in ("checkpoint", "temperature", "thresholds_from_val")}, indent=1))
    for split in ("dev", "heldback"):
        if split in summary:
            print(split, "vote_ex", summary[split]["vote_ex"], "auroc", summary[split]["auroc"],
                  summary[split]["at_val_thresholds"])


if __name__ == "__main__":
    main(sys.argv[1])
