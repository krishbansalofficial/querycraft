"""Evaluation gate (v2 part 5): CI fails if execution accuracy on a fixed 200 question val slice drops.

CI has no GPU and no BIRD databases, so the gate replays a cached model: every completion is stored by the hash of
its exact prompt, and every SQL result (predicted and gold) is stored by the hash of the query. In CI the current
code rebuilds each prompt (schema serialization, pruning, linked tables, chat template), looks the completion up,
extracts the SQL and scores it against the stored gold result with the current matching rules. A change to prompts,
extraction or scoring therefore moves the score; a prompt the cache has never seen counts as wrong.

    python -m querycraft.gate check                     # CPU, CI: exit 1 if EX < baseline - margin
    python -m querycraft.gate refresh configs/retriever/70_qwen1.5b_lora_r16_retrieved.yaml   # GPU, laptop

Refreshing the baseline (after an intended prompt or model change): run `make gate-refresh` on a machine with the
GPU stack and the BIRD databases, review the EX change it prints, and commit data/gate/. The margin is documented in
data/gate/baseline.json: 1.0 point (2 of 200 questions). The replay is deterministic, so any drop is caused by a
code change; the margin only lets a deliberate one question change through without a GPU refresh.
"""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

GATE = Path("data/gate")
SLICE, CACHE, RESULTS, BASELINE = GATE / "val_slice.jsonl", GATE / "completions.jsonl", GATE / "results.jsonl", GATE / "baseline.json"
SCHEMAS = GATE / "schemas"
N, SEED, MARGIN = 200, 0, 1.0
TOKENIZER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines()] if Path(p).exists() else []


def _gate_rows():
    """Slice rows pointed at the committed schema caches (read_schema finds <db>.schema3.json next to db_path)."""
    return [{**r, "db_path": str(SCHEMAS / r["db_id"] / f"{r['db_id']}.sqlite")} for r in _jsonl(SLICE)]


def _prompts(tok, rows):
    from .prompts import PromptBuilder
    pb = PromptBuilder(tok, 1280, linking="retriever")
    return [pb.prompt_text(r) for r in rows]


def result_hashes(rows):
    """Same equality as execution.results_match: ordered list, or multiset of rows (repr keeps types apart)."""
    from collections import Counter
    reps = [repr(tuple(r)) for r in rows]
    return {"ordered": sha(json.dumps(reps)), "multiset": sha(json.dumps(sorted(Counter(reps).items())))}


def score(rows, prompts, cache, results):
    from .execution import has_top_level_order_by
    from .prompts import extract_sql
    hits, right = 0, 0
    per = []
    for r, p in zip(rows, prompts):
        comp = cache.get(sha(p))
        ok = False
        if comp is not None:
            hits += 1
            pred, gold = results.get(sha(extract_sql(comp))), results[sha(r["sql"])]
            if pred and pred["error"] is None and gold["error"] is None:
                key = "ordered" if has_top_level_order_by(r["sql"]) else "multiset"
                ok = pred[key] == gold[key]
        right += ok
        per.append({"id": r["id"], "ex": ok, "cached": comp is not None})
    return {"n": len(rows), "ex": round(100 * right / len(rows), 2), "cache_hits": hits}, per


def check():
    from transformers import AutoTokenizer
    base = json.loads(BASELINE.read_text())
    rows = _gate_rows()
    cache = {d["prompt"]: d["completion"] for d in _jsonl(CACHE)}
    results = {d["sql"]: d for d in _jsonl(RESULTS)}
    m, _ = score(rows, _prompts(AutoTokenizer.from_pretrained(TOKENIZER), rows), cache, results)
    drop = round(base["ex"] - m["ex"], 2)
    print(json.dumps({"baseline_ex": base["ex"], "ex": m["ex"], "drop": drop, "margin": base["margin"],
                      "cache_hits": m["cache_hits"], "n": m["n"]}, indent=1))
    if drop > base["margin"]:
        print(f"GATE FAILED: EX fell {drop} points (> {base['margin']}). If intended, run `make gate-refresh`.")
        sys.exit(1)
    print("gate passed")


def refresh(cfg_path):
    """GPU + BIRD databases: rebuild the slice fixtures, generate, execute, and save the baseline."""
    from .config import load_config
    from .data import load_split, selection
    from .execution import execute
    from .infer import Generator
    from .schema import read_schema

    @selection  # val only
    def build_slice():
        val = load_split("val")
        rows = sorted(random.Random(SEED).sample(val, N), key=lambda r: r["id"])
        tables = {(d := json.loads(l))["id"]: d["tables"] for l in
                  Path("runs/table_retriever/eval/val_tables.jsonl").read_text().splitlines()}
        return [{"id": r["id"], "db_id": r["db_id"], "question": r["question"], "evidence": r.get("evidence", ""),
                 "sql": r["sql"], "difficulty": r.get("difficulty"), "link_tables": tables[r["id"]],
                 "real_db_path": r["db_path"]} for r in rows]

    cfg = load_config(cfg_path)
    rows = build_slice()
    SCHEMAS.mkdir(parents=True, exist_ok=True)
    for db, path in {r["db_id"]: r["real_db_path"] for r in rows}.items():
        (SCHEMAS / db).mkdir(exist_ok=True)
        (SCHEMAS / db / f"{db}.schema3.json").write_text(json.dumps(read_schema(path)))
    SLICE.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "real_db_path"}) + "\n" for r in rows))
    gen = Generator(cfg, cfg["model"])
    gate_rows = _gate_rows()
    prompts = _prompts(gen.tok, gate_rows)
    comps = gen.generate(prompts)
    CACHE.write_text("".join(json.dumps({"prompt": sha(p), "completion": c}) + "\n" for p, c in zip(prompts, comps)))
    from .prompts import extract_sql
    res = {}
    for r, c in zip(rows, comps):
        for q in (r["sql"], extract_sql(c)):
            if sha(q) not in res:
                out, err = execute(r["real_db_path"], q, 60.0)
                res[sha(q)] = {"sql": sha(q), "error": err, **(result_hashes(out) if err is None else {})}
    RESULTS.write_text("".join(json.dumps(v) + "\n" for v in res.values()))
    cache = {sha(p): c for p, c in zip(prompts, comps)}
    m, _ = score(gate_rows, prompts, cache, res)
    BASELINE.write_text(json.dumps({**m, "margin": MARGIN, "model": cfg["model"], "config": cfg_path,
                                    "refreshed": time.strftime("%Y-%m-%d"), "seed": SEED,
                                    "note": "EX on 200 seeded val questions; replayed in CI from cached completions"},
                                   indent=2))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    check() if sys.argv[1] == "check" else refresh(sys.argv[2])
