"""Evaluation pipeline for one config.

  1. Trained runs: generate on val for every candidate checkpoint, pick the best val EX.
     (BIRD dev is never looked at for any choice.)
  2. Generate on dev + heldback (+ forgetting probe) with the chosen checkpoint (or the base model).
  3. Score by execution and write runs/<name>/results.json.

Each vLLM engine runs in its own subprocess so GPU memory is fully released between models.

    python -m querycraft.run_eval configs/local/qwen1.5b_qlora_r16.yaml
"""
import json
import subprocess
import sys
from pathlib import Path

from .config import load_config, logged_stage, run_dir
from .evaluate import score_file
from .infer import is_adapter, is_dora, merge_adapter


def _engine(cfg_path, model_path, jobs):
    """jobs: list of {kind: split|forgetting, split, adapter, out}. Skips jobs whose output exists."""
    jobs = [j for j in jobs if not Path(j["out"]).exists()]
    if jobs:
        subprocess.run([sys.executable, "-m", "querycraft.run_eval", cfg_path, "--engine", model_path or "",
                        json.dumps(jobs)], check=True)


def _engine_main(cfg_path, model_path, jobs):
    from . import forgetting
    from .infer import Generator, predict
    cfg = load_config(cfg_path)
    gen = Generator(cfg, model_path or None)
    for j in jobs:
        if j["kind"] == "forgetting":
            res = forgetting.run(gen, j.get("adapter"), out_path=j["out"] + "l")
            Path(j["out"]).write_text(json.dumps(res, indent=2))
        else:
            predict(gen, cfg, j["split"], j.get("adapter"), j["out"])


def _serving(cfg, ckpt):
    """(model_path, adapter) for vLLM."""
    if ckpt is None:
        return None, None
    if is_adapter(ckpt):
        return (merge_adapter(cfg["model"], ckpt), None) if is_dora(ckpt) else (None, str(ckpt))
    return str(ckpt), None  # full fine tune checkpoint


def main(cfg_path):
    cfg = load_config(cfg_path)
    ev_dir = run_dir(cfg) / "eval"
    ev_dir.mkdir(exist_ok=True)
    results = {"name": cfg["name"], "model": cfg["model"], "method": cfg.get("train", {}).get("method", "base"),
               "shots": cfg["prompt"].get("shots", 0)}
    with logged_stage(cfg, "eval") as meta:
        chosen = None
        if "train" in cfg:
            ckpts = [run_dir(cfg) / "ckpt" / f"checkpoint-{s}" for s in cfg["train"]["select_steps"]]
            missing = [str(c) for c in ckpts if not c.exists()]
            if missing:
                raise FileNotFoundError(f"train first, missing {missing}")
            val = {}
            adapters = all(is_adapter(c) and not is_dora(c) for c in ckpts)
            if adapters:  # one engine, many adapters
                _engine(cfg_path, None, [{"kind": "split", "split": "val", "adapter": str(c),
                                          "out": str(ev_dir / f"val_{c.name}.jsonl")} for c in ckpts])
            else:
                for c in ckpts:
                    mp, ad = _serving(cfg, c)
                    _engine(cfg_path, mp, [{"kind": "split", "split": "val", "adapter": ad,
                                            "out": str(ev_dir / f"val_{c.name}.jsonl")}])
            for c in ckpts:
                val[c.name] = score_file(ev_dir / f"val_{c.name}.jsonl")["ex"]
            chosen = max(ckpts, key=lambda c: (val[c.name], int(c.name.split("-")[1])))  # ties -> later step
            results["val_ex_by_checkpoint"] = val
            results["selected_checkpoint"] = chosen.name
        model_path, adapter = _serving(cfg, chosen)
        tag = chosen.name if chosen else "base"
        jobs = [{"kind": "split", "split": s, "adapter": adapter, "out": str(ev_dir / f"{s}_{tag}.jsonl")}
                for s in cfg["eval"]["splits"]]
        if cfg["eval"].get("forgetting"):
            jobs.append({"kind": "forgetting", "adapter": adapter, "out": str(ev_dir / f"forgetting_{tag}.json")})
        _engine(cfg_path, model_path, jobs)
        for s in cfg["eval"]["splits"]:
            results[s] = score_file(ev_dir / f"{s}_{tag}.jsonl")
        if cfg["eval"].get("forgetting"):
            results["forgetting"] = json.loads((ev_dir / f"forgetting_{tag}.json").read_text())
        (run_dir(cfg) / "results.json").write_text(json.dumps(results, indent=2))
        meta["results"] = "results.json"
    print(json.dumps({k: v for k, v in results.items() if k not in ("val", "dev", "heldback")}, indent=2))
    for s in cfg["eval"]["splits"]:
        print(s, results[s]["ex"], results[s]["by_difficulty"])


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--engine":
        _engine_main(sys.argv[1], sys.argv[3], json.loads(sys.argv[4]))
    else:
        main(sys.argv[1])
