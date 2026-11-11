"""Prompt token lengths before vs after schema pruning, and pruning recall
(share of examples whose gold SQL tables all survive pruning). Writes data/processed/prompt_stats.json."""
import json
import sys

import numpy as np
from transformers import AutoTokenizer

from .config import load_config
from .data import PROC, load_split
from .prompts import PromptBuilder, gold_tables
from .schema import prune, read_schema


def _summ(xs):
    xs = np.array(xs)
    return {"mean": round(float(xs.mean()), 1), "p50": int(np.percentile(xs, 50)), "p95": int(np.percentile(xs, 95)),
            "max": int(xs.max())}


def main(cfg_path, budget=None, splits=("train", "val", "dev", "heldback")):
    cfg = load_config(cfg_path)
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    pb = PromptBuilder(tok, int(budget or cfg["prompt"]["schema_budget"]))
    out = {"tokenizer": cfg["model"], "schema_budget": pb.budget}
    for split in splits:
        rows = load_split(split)
        if not rows:
            continue
        before, after, recall = [], [], []
        for ex in rows:
            before.append(pb.n_tokens(pb.prompt_text(ex, pruned=False)))
            after.append(pb.n_tokens(pb.prompt_text(ex, pruned=True)))
            keep, _ = prune(read_schema(ex["db_path"]), ex["question"], ex["evidence"], pb.n_tokens, pb.budget)
            recall.append(gold_tables(ex["sql"]) <= {t.lower() for t in keep})
        out[split] = {"n": len(rows), "before": _summ(before), "after": _summ(after),
                      "pruned_share": round(float(np.mean([a < b for a, b in zip(after, before)])), 4),
                      "gold_table_recall": round(float(np.mean(recall)), 4)}
        print(split, json.dumps(out[split]))
    if budget is None:  # only the configured budget is the official record
        (PROC / "prompt_stats.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]), *([sys.argv[3].split(",")] if len(sys.argv) > 3 else []))
