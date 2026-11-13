"""GRPO step 2: keep train questions the SFT model solves sometimes but not always (docs/plans/grpo.md).

Samples N queries per question at temperature 1.0 and keeps questions with 1 to N-1 correct: groups that are
all right or all wrong give GRPO zero advantage, so zero learning signal. Train only; val/dev/held back untouched.

    python -m querycraft.grpo_data configs/local/11_qwen1.5b_lora_r16.yaml
"""
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from .config import load_config, logged_stage, run_dir
from .reward import _gold, binary_one

N, TEMP, POOL, CAP, SEED = 8, 1.0, 6000, 2000, 0  # ponytail: a 6,000 question sample of train, not all of it


def _count(item):
    if _gold(item["db_path"], item["sql"])[1] is not None:
        return None  # broken gold query: no reward possible
    return sum(binary_one(item["db_path"], item["sql"], s) for s in item["samples"])


def main(cfg_path):
    from .data import load_split
    from .infer import Generator
    from .prompts import PromptBuilder, extract_sql
    from .run_eval import _serving

    cfg = load_config(cfg_path)
    tag = json.loads((run_dir(cfg) / "results.json").read_text())["selected_checkpoint"]
    out = run_dir(cfg) / "grpo"
    out.mkdir(exist_ok=True)
    rows = load_split("train")
    random.Random(SEED).shuffle(rows)
    rows = rows[:POOL]
    with logged_stage(cfg, "grpo_data") as meta:
        cache = out / "train_samples.jsonl"
        if cache.exists():
            samples = [json.loads(l)["samples"] for l in cache.read_text().splitlines()]
        else:
            model_path, adapter = _serving(cfg, run_dir(cfg) / "ckpt" / tag)
            gen = Generator(cfg, model_path)
            pb = PromptBuilder(gen.tok, cfg["prompt"]["schema_budget"])
            samples = [[extract_sql(s) for s in ss] for ss in
                       gen.generate([pb.prompt_text(r) for r in rows], adapter, n=N, temperature=TEMP)]
            cache.write_text("".join(json.dumps({"id": r["id"], "samples": s}) + "\n" for r, s in zip(rows, samples)))
        items = [r | {"samples": s} for r, s in zip(rows, samples)]
        with ProcessPoolExecutor(8, mp_context=get_context("spawn")) as pool:
            counts = list(pool.map(_count, items, chunksize=4))
        hist = {k: counts.count(k) for k in [None, *range(N + 1)]}
        keep = [r for r, k in zip(rows, counts) if k is not None and 0 < k < N][:CAP]  # rows already shuffled
        (out / "prompts.jsonl").write_text("".join(
            json.dumps({"id": r["id"], "db_id": r["db_id"], "db_path": r["db_path"], "gold_sql": r["sql"]}) + "\n"
            for r in keep))
        meta |= {"checkpoint": tag, "pool": len(rows), "kept": len(keep), "correct_of_8_histogram": hist}
    print(json.dumps({"kept": len(keep), "histogram": hist}))


if __name__ == "__main__":
    main(sys.argv[1])
