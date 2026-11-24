"""Learned answer or abstain model (v2 part 2): does a calibrated classifier beat the vote share threshold?

The answers are the existing vote of 8 answers of the 1.5B LoRA r16 SFT model (querycraft/confidence.py, T=0.6
chosen on val in v1), so the baseline and the learned model decide over exactly the same SQL; only the score that
ranks questions differs. Label: the voted answer is correct (execution match).

    python -m querycraft.abstain configs/agent/abstain.yaml

fit() sees val only (5 fold CV picks the LightGBM settings and the calibration method; out of fold predictions
pick the coverage thresholds). report() then scores dev and held back once.
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import sqlglot
from sqlglot import exp

from .confidence import auroc, selective
from .config import load_config, logged_stage, run_dir
from .data import load_split, selection
from .prompts import gold_tables
from .schema import read_schema

FEATURES = ["vote_share", "n_results", "greedy_error", "vote_is_greedy", "ret_top", "ret_margin", "ret_kth",
            "n_tables", "n_joins", "question_words", "evidence_words", "db_tables", "db_log_mb"]
BASELINE = "vote_share"


def _jsonl(p):
    return {(d := json.loads(l))["id"]: d for l in Path(p).read_text().splitlines()}


def _joins(sql):
    try:
        return len(list(sqlglot.parse_one(sql, read="sqlite").find_all(exp.Join)))
    except Exception:
        return 0


def features(cfg, split):
    """One row per question with a judgeable gold query: FEATURES + label + id."""
    votes = _jsonl(cfg["votes"].format(split=split))
    greedy = _jsonl(cfg["greedy"].format(split=split))
    ranked = _jsonl(cfg["ranked"].format(split=split))
    chosen = json.loads(Path(cfg["retriever_sweep"]).read_text())["chosen"]  # source and k picked on val
    src, k = chosen["source"], chosen["k"]
    rows = []
    for ex in load_split(split):
        v = votes.get(ex["id"])
        if v is None or v["gold_error"]:
            continue
        rows.append({"id": ex["id"], "label": int(v["ex"]),
                     **feature_row(ex, v, greedy[ex["id"]], [s for _, s in ranked[ex["id"]][src]], k)})
    return rows


def feature_row(ex, vote, greedy, scores, k):
    """vote: {confidence, n_results, chosen_sql}; greedy: {pred_sql, pred_error}; scores: retriever scores, ranked.
    Shared by training (above) and the live service (serve/app.py), so both see identical features."""
    return {
        "vote_share": vote["confidence"], "n_results": vote["n_results"],
        "greedy_error": int(greedy["pred_error"] is not None), "vote_is_greedy": int(vote["chosen_sql"] == greedy["pred_sql"]),
        "ret_top": scores[0] if scores else 0.0, "ret_margin": (scores[0] - scores[1]) if len(scores) > 1 else 0.0,
        "ret_kth": scores[min(k, len(scores)) - 1] if scores else 0.0,
        "n_tables": len(gold_tables(vote["chosen_sql"])), "n_joins": _joins(vote["chosen_sql"]),
        "question_words": len(ex["question"].split()), "evidence_words": len((ex.get("evidence") or "").split()),
        "db_tables": len(read_schema(ex["db_path"])),
        "db_log_mb": math.log1p(os.path.getsize(ex["db_path"]) / 2**20),
    }


def _xy(rows):
    import numpy as np
    return np.array([[r[f] for f in FEATURES] for r in rows], dtype=float), np.array([r["label"] for r in rows])


def brier(p, y):
    return round(sum((a - b) ** 2 for a, b in zip(p, y)) / len(y), 4)


def ece(p, y, bins=10):
    """Expected calibration error, equal width bins."""
    tot = 0.0
    for b in range(bins):
        idx = [i for i, x in enumerate(p) if (b / bins <= x < (b + 1) / bins) or (b == bins - 1 and x == 1.0)]
        if idx:
            tot += len(idx) / len(p) * abs(sum(p[i] for i in idx) / len(idx) - sum(y[i] for i in idx) / len(idx))
    return round(tot, 4)


def _lgbm(params, seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**params, random_state=seed, min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, verbose=-1)


@selection
def fit(cfg):
    """Everything chosen on val: LightGBM settings and calibration (5 fold CV Brier), coverage thresholds
    (out of fold calibrated predictions)."""
    import itertools

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    rows = features(cfg, "val")
    X, y = _xy(rows)
    cv = StratifiedKFold(5, shuffle=True, random_state=cfg["seed"])
    grid = cfg["grid"]
    cands = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    scores = {}
    for params in cands:
        for method in cfg["calibration"]:
            m = CalibratedClassifierCV(_lgbm(params, cfg["seed"]), method=method, cv=5)
            p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
            scores[json.dumps([params, method])] = brier(p.tolist(), y.tolist())
    params, method = json.loads(min(scores, key=scores.get))
    model = CalibratedClassifierCV(_lgbm(params, cfg["seed"]), method=method, cv=5).fit(X, y)
    oof = cross_val_predict(CalibratedClassifierCV(_lgbm(params, cfg["seed"]), method=method, cv=5), X, y, cv=cv,
                            method="predict_proba")[:, 1].tolist()
    from .confidence import thresholds_for_coverage
    return model, {"params": params, "calibration": method, "cv_brier": scores, "n_val": len(rows),
                   "val_oof_brier": brier(oof, y.tolist()), "val_oof_auroc": auroc(oof, y.tolist()),
                   "val_oof": [round(x, 5) for x in oof],  # reference distribution for the drift report
                   "thresholds": thresholds_for_coverage(oof, cfg["coverage"]),
                   "baseline_thresholds": thresholds_for_coverage([r[BASELINE] for r in rows], cfg["coverage"])}


def at_coverage(score, y, cov, seed=0):
    """Answer exactly round(cov * n) questions, highest score first; ties broken by a seeded shuffle."""
    rng = random.Random(seed)
    order = sorted(range(len(y)), key=lambda i: (-score[i], rng.random()))
    return set(order[:round(cov * len(y))])


def sign_p(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def compare(learned, base, y, cov, n_boot=2000, seed=0):
    """Both answer the same number of questions. Paired sign test over questions answered correctly by exactly one
    of them, plus a paired bootstrap 95% interval for the accuracy difference (learned - baseline)."""
    A, B = at_coverage(learned, y, cov, seed), at_coverage(base, y, cov, seed)
    a_right = [i for i in A if y[i]]
    b_right = [i for i in B if y[i]]
    only_a = len(set(a_right) - set(b_right))
    only_b = len(set(b_right) - set(a_right))
    rng = random.Random(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        s = [rng.randrange(n) for _ in range(n)]
        ys, ls, bs = [y[i] for i in s], [learned[i] for i in s], [base[i] for i in s]
        a2, b2 = at_coverage(ls, ys, cov, seed), at_coverage(bs, ys, cov, seed)
        diffs.append(100 * (sum(ys[i] for i in a2) - sum(ys[i] for i in b2)) / max(1, len(a2)))
    diffs.sort()
    return {"coverage": cov, "answered": len(A),
            "learned_acc": round(100 * len(a_right) / len(A), 2), "baseline_acc": round(100 * len(b_right) / len(B), 2),
            "only_learned_right": only_a, "only_baseline_right": only_b, "p": round(sign_p(only_a, only_b), 4),
            "diff_ci95": [round(diffs[int(0.025 * n_boot)], 2), round(diffs[int(0.975 * n_boot) - 1], 2)]}


def curve(score, y, points=60):
    ts = sorted(set(score))
    ts = [ts[int(i * (len(ts) - 1) / (points - 1))] for i in range(points)] if len(ts) > points else ts
    return [selective(score, y, t) for t in ts]


def report(cfg, model, chosen, split):
    rows = features(cfg, split)
    X, y = _xy(rows)
    y = y.tolist()
    p = model.predict_proba(X)[:, 1].tolist()
    b = [r[BASELINE] for r in rows]
    return {"n": len(rows), "learned": {"brier": brier(p, y), "ece": ece(p, y), "auroc": auroc(p, y), "curve": curve(p, y),
                                        "at_val_thresholds": {k: selective(p, y, t) for k, t in chosen["thresholds"].items()}},
            "baseline": {"brier": brier(b, y), "ece": ece(b, y), "auroc": auroc(b, y), "curve": curve(b, y),
                         "at_val_thresholds": {k: selective(b, y, t) for k, t in chosen["baseline_thresholds"].items()}},
            "at_70": compare(p, b, y, 0.7), "scores": [{"id": r["id"], "p": round(x, 5)} for r, x in zip(rows, p)]}


def main(cfg_path):
    import joblib
    cfg = load_config(cfg_path)
    out = run_dir(cfg)
    with logged_stage(cfg, "fit") as meta:
        model, chosen = fit(cfg)
        joblib.dump(model, out / "model.joblib")
        res = {"name": cfg["name"], "features": FEATURES, "fit": chosen}
        for split in ("dev", "heldback"):
            r = report(cfg, model, chosen, split)
            (out / f"{split}_scores.jsonl").write_text("".join(json.dumps(s) + "\n" for s in r.pop("scores")))
            res[split] = r
        (out / "results.json").write_text(json.dumps(res, indent=2))
        meta["results"] = "results.json"
    d = res["dev"]
    print(json.dumps({"fit": {k: chosen[k] for k in ("params", "calibration", "val_oof_brier", "thresholds")},
                      "dev_learned": {k: d["learned"][k] for k in ("brier", "ece", "auroc")},
                      "dev_baseline": {k: d["baseline"][k] for k in ("brier", "ece", "auroc")},
                      "dev_at_70": d["at_70"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
