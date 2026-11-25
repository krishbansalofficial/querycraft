"""Serving benchmark (v2 part 4): 1.5B in bf16 (base + LoRA adapter) against 4 bit AWQ (W4A16, merged SFT model).

    python -m querycraft.bench configs/serve/bench.yaml

For each variant: start `vllm serve`, record model memory and KV cache size from its log and the peak GPU memory
from nvidia-smi, then load it with Locust at each concurrency for a fixed time. Accuracy is measured separately by
the normal evaluation (configs/serve/81_*: AWQ on val, dev, held back), and paired against the bf16 run on the same
dev questions in the report. Cost per 1,000 queries = GPU price per hour / (3,600 x requests per second) x 1,000,
at public on demand prices (modal.com/pricing, read 2026-11-23), assuming the throughput measured on this GPU.
"""
import csv
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import load_config, logged_stage, run_dir


def write_prompts(cfg, out):
    """The evaluation's exact single pass prompts for dev (v1 prompt, full pruned schema)."""
    from transformers import AutoTokenizer

    from .data import load_split
    from .prompts import PromptBuilder
    pb = PromptBuilder(AutoTokenizer.from_pretrained(cfg["tokenizer"]), 1280)
    rows = load_split("dev")
    out.write_text("".join(json.dumps({"id": r["id"], "prompt": pb.prompt_text(r)}) + "\n" for r in rows))


class GpuPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, False

    def run(self):
        while not self.stop:
            try:
                used = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                   text=True).split()[0])
                self.peak = max(self.peak, used)
            except Exception:
                pass
            time.sleep(0.5)


def _wait(url, proc, timeout=900):
    import httpx
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError("vllm serve exited; see its log")
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("vllm serve did not become healthy")


def _log_numbers(text):
    weights = re.findall(r"Model loading took ([\d.]+) GiB", text)
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    return {"model_memory_gib": float(weights[-1]) if weights else None,
            "kv_cache_tokens": int(kv[-1].replace(",", "")) if kv else None}


def load_test(cfg, url, model, prompts, c, out):
    stem = out / f"c{c}"
    env = dict(os.environ, QC_PROMPTS=str(prompts), QC_MODEL=model, QC_TOKENS_OUT=f"{stem}_tokens.json")
    locust = Path(sys.executable).with_name("locust")
    subprocess.run([str(locust), "-f", "serve/bench/locustfile.py", "--headless", "-u", str(c), "-r", str(c),
                    "-t", f"{cfg['seconds']}s", "--host", url, "--csv", str(stem), "--only-summary"],
                   env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    row = next(r for r in csv.DictReader(open(f"{stem}_stats.csv")) if r["Name"] == "completion")
    tok = json.loads(Path(f"{stem}_tokens.json").read_text())
    n, fails = int(row["Request Count"]), int(row["Failure Count"])
    secs = cfg["seconds"]
    return {"concurrency": c, "requests": n, "failures": fails, "requests_per_s": round((n - fails) / secs, 3),
            "completion_tokens_per_s": round(tok["completion"] / secs, 1),
            "p50_ms": float(row["50%"]), "p99_ms": float(row["99%"]), "mean_ms": round(float(row["Average Response Time"]), 1)}


def bench_variant(cfg, v, prompts, out):
    out.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{cfg['port']}"
    log = open(out / "vllm.log", "w")
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")
    vllm = Path(sys.executable).with_name("vllm")
    cmd = [str(vllm), "serve", v["model"], "--port", str(cfg["port"]), "--max-model-len", "4096",
           "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]), "--seed", "0", "--served-model-name", v["name"]]
    if v.get("adapter"):
        cmd = cmd[:-2] + ["--served-model-name", "base", "--enable-lora", "--max-lora-rank", "64",
                          "--lora-modules", f"{v['name']}={v['adapter']}"]
    cmd += v.get("args", [])   # extra vllm serve flags, e.g. the memory limits the 7B needs on 8 GB
    peak = GpuPeak()
    peak.start()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        startup = _wait(url, proc)
        levels = [load_test(cfg, url, v["name"], prompts, c, out) for c in cfg["concurrency"]]
    finally:
        proc.terminate()
        proc.wait(60)
        peak.stop = True
        log.close()
    return {"variant": v["name"], "model": v["model"], "adapter": v.get("adapter"), "startup_s": round(startup, 1),
            "peak_gpu_mib": peak.peak, **_log_numbers((out / "vllm.log").read_text()), "levels": levels,
            "cost_per_1000_queries_usd": {gpu: {str(l["concurrency"]): round(price / (3600 * l["requests_per_s"]) * 1000, 4)
                                                for l in levels if l["requests_per_s"]}
                                          for gpu, price in cfg["gpu_price_per_hour"].items()}}


def main(cfg_path):
    cfg = load_config(cfg_path)
    out = run_dir(cfg)
    prompts = out / "prompts.jsonl"
    if not prompts.exists():
        write_prompts(cfg, prompts)
    with logged_stage(cfg, "bench") as meta:
        res = {"name": cfg["name"], "seconds_per_level": cfg["seconds"], "gpu_price_per_hour": cfg["gpu_price_per_hour"],
               "variants": [bench_variant(cfg, v, prompts, out / v["name"]) for v in cfg["variants"]]}
        (out / "results.json").write_text(json.dumps(res, indent=2))
        meta["results"] = "results.json"
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
