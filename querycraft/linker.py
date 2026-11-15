"""Schema linking step 3: evaluate the trained table selector (docs/plans/schema_linking.md).

    python -m querycraft.train  configs/selector/60_qwen1.5b_table_selector.yaml
    python -m querycraft.linker configs/selector/60_qwen1.5b_table_selector.yaml

Picks the checkpoint with the best val exact set match (dev is never used for a choice), then writes
runs/<name>/eval/{split}_tables.jsonl ({id, tables}) for the second pass and results.json with set metrics.
"""
import json
import re
import sys

from .config import load_config, logged_stage, run_dir
from .data import load_split
from .prompts import LINK_SYSTEM, PromptBuilder, gold_tables
from .schema import read_schema


def parse_tables(text, db_path):
    """Comma separated names -> the subset that exists in the database (lower case)."""
    names = {t["name"].lower() for t in read_schema(db_path)}
    return sorted({w.strip(" `\"'.").lower() for w in re.split(r"[,\n]", text)} & names)


def set_metrics(rows):
    """rows: [(pred set, gold set)] -> recall, precision, exact match, every gold table kept, mean predicted."""
    hit = sum(len(p & g) for p, g in rows)
    return {"recall": round(hit / max(1, sum(len(g) for _, g in rows)), 4),
            "precision": round(hit / max(1, sum(len(p) for p, _ in rows)), 4),
            "exact": round(sum(p == g for p, g in rows) / len(rows), 4),
            "all_gold_kept": round(sum(g <= p for p, g in rows) / len(rows), 4),
            "mean_predicted": round(sum(len(p) for p, _ in rows) / len(rows), 2)}


def _run(gen, cfg, split, adapter):
    rows = load_split(split)
    pb = PromptBuilder(gen.tok, cfg["prompt"]["schema_budget"])
    raw = gen.generate([pb.prompt_text(ex, system=LINK_SYSTEM) for ex in rows], adapter, max_tokens=64)
    preds = [{"id": ex["id"], "tables": parse_tables(r, ex["db_path"]), "raw": r} for ex, r in zip(rows, raw)]
    m = set_metrics([(set(p["tables"]), gold_tables(ex["sql"])) for p, ex in zip(preds, rows)])
    return preds, m


def main(cfg_path):
    from .infer import Generator
    cfg = load_config(cfg_path)
    ev = run_dir(cfg) / "eval"
    ev.mkdir(exist_ok=True)
    with logged_stage(cfg, "eval") as meta:
        gen = Generator(cfg)
        ckpts = [run_dir(cfg) / "ckpt" / f"checkpoint-{s}" for s in cfg["train"]["select_steps"]]
        val = {c.name: _run(gen, cfg, "val", str(c))[1] for c in ckpts}
        chosen = max(ckpts, key=lambda c: (val[c.name]["exact"], int(c.name.split("-")[1])))
        res = {"name": cfg["name"], "val_by_checkpoint": val, "selected_checkpoint": chosen.name}
        for split in ("val", "dev", "heldback"):
            preds, res[split] = _run(gen, cfg, split, str(chosen))
            (ev / f"{split}_tables.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
        (run_dir(cfg) / "results.json").write_text(json.dumps(res, indent=2))
        meta["results"] = "results.json"
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
