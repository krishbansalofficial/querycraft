"""Run the rented profile on Modal's free monthly credit, guarded by querycraft.budget.

    modal run scripts/modal_app.py::prepare      # CPU only: BIRD onto the volume, splits, schema cache, tests
    modal run scripts/modal_app.py --smoke       # 0.5B rehearsal on an L4 (a few cents)
    modal run scripts/modal_app.py --dry-run     # print which jobs this billing cycle's budget allows
    modal run --detach scripts/modal_app.py      # run them; jobs that do not fit wait for the next cycle
    modal volume get querycraft-vol runs/<name> runs/   # copy results back for `make report`

Guardrails: a job starts only if its worst case cost fits under the cap (no run is ever cut off halfway),
DoRA waits until every sweep run it picks its rank from is finished, and every job's real cost is appended
to /vol/modal_ledger.json. The queue loop itself runs on Modal, so closing the laptop does not stop it.
"""
import subprocess
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
VLLM, TORCH = "0.32.0", "2.13.0"
ORDER = [  # what the laptop cannot do first; DoRA last because it needs the whole sweep
    "30_qwen1.5b_full.yaml", "01_qwen7b_zeroshot.yaml", "11_qwen7b_qlora_r16.yaml", "10_qwen7b_qlora_r8.yaml",
    "12_qwen7b_qlora_r64.yaml", "02_qwen7b_fewshot.yaml", "20_qwen7b_dora_best.yaml"]
SMOKE_ORDER = ["01_smoke_zeroshot.yaml", "10_smoke_qlora_r8.yaml", "11_smoke_qlora_r16.yaml",
               "20_smoke_dora_best.yaml", "30_smoke_full.yaml"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "build-essential")  # Triton compiles a small C helper at runtime
    .pip_install("uv")
    .add_local_file(REPO / "requirements.txt", "/tmp/requirements.txt", copy=True)
    .run_commands(
        "uv pip install --system --index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cu129 "
        f"'torch=={TORCH}+cu129' "
        f"'vllm @ https://github.com/vllm-project/vllm/releases/download/v{VLLM}/vllm-{VLLM}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' "
        "-r /tmp/requirements.txt",
        "uv pip uninstall --system torchcodec")
    .env({"DATA_DIR": "/vol/data", "HF_HOME": "/vol/hf", "PYTHONPATH": "/root/querycraft",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(REPO, "/root/querycraft", ignore=[".venv", ".git", "runs", "runs_*", "data/raw",
                                                      "data/processed", "session_logs", "wandb", "docs",
                                                      "**/__pycache__", "modal_ledger.json"]))
vol = modal.Volume.from_name("querycraft-vol", create_if_missing=True)
app = modal.App("querycraft", image=image, volumes={"/vol": vol})


def _sh(cmd, env=None):
    print("==", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd="/root/querycraft", env=env)


@app.function(cpu=2, memory=8192, timeout=4 * 3600)
def prepare():
    _sh("bash scripts/download_bird.sh && python -m querycraft.data && "
        "python -m querycraft.prompt_stats configs/local/01_qwen1.5b_zeroshot.yaml && python -m pytest -q tests")
    vol.commit()


def _job(cfg, runs, git):
    import os
    env = dict(os.environ, RUNS_DIR=f"/vol/{runs}", QC_GIT_COMMIT=git)
    try:
        if "/0" not in cfg:  # configs starting with 0 are baselines (no training)
            _sh(f"python -m querycraft.train {cfg}", env)
        _sh(f"python -m querycraft.run_eval {cfg}", env)
    finally:
        vol.commit()


@app.function(gpu="A100-80GB", cpu=4, memory=32768, timeout=10 * 3600)
def job(cfg, runs, git):
    _job(cfg, runs, git)


@app.function(gpu="L4", cpu=4, memory=16384, timeout=3600)
def smoke_job(cfg, runs, git):
    _job(cfg, runs, git)


def _wait(call_id):
    """A single blocking .remote() died after ~1.5 h with 'AuthError: Jwt is expired', which cancelled the job.
    Poll the spawned call in 10 minute slices instead, re-fetching it (fresh token) on every slice."""
    while True:
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=600)
        except (TimeoutError, modal.exception.TimeoutError, modal.exception.AuthError) as ex:
            if isinstance(ex, modal.exception.FunctionTimeoutError):
                raise  # the job itself hit its 10 h limit
            continue  # job still running; a job's own failure is any other exception and propagates


@app.function(cpu=0.25, memory=1024, timeout=24 * 3600)
def queue(smoke, dry_run, git, allow_partial=False):
    """The job loop runs on Modal (a few cents of CPU), so it survives the laptop sleeping."""
    import sys
    sys.path.insert(0, "/root/querycraft")
    from querycraft import budget
    from querycraft.config import load_config

    profile, runs, order, fn = ("smoke", "runs_smoke", SMOKE_ORDER, smoke_job) if smoke else \
                               ("rented", "runs", ORDER, job)
    ledger = "/vol/modal_ledger.json"
    vol.reload()
    done = {p.parent.name for p in Path(f"/vol/{runs}").glob("*/results.json")}
    for f in order:
        cfg = load_config(Path("/root/querycraft/configs") / profile / f)
        name, rows = cfg["name"], budget.load(ledger)
        if name in done:
            print(f"done     {name}")
            continue
        missing = [r for r in cfg.get("train", {}).get("rank_from", []) if r not in done]
        if missing:
            print(f"wait     {name}: needs {missing} first")
            continue
        ok, worst = budget.can_start(rows, cfg.get("budget_hours", 0.3), allow_partial=allow_partial)
        if not ok:
            print(f"defer    {name}: worst case ${worst} + spent ${budget.spent(rows)} > cap "
                  f"${budget.CREDIT_USD if allow_partial else budget.CAP_USD}")
            continue
        print(f"{'would run' if dry_run else 'run'}  {name}: worst case ${worst}, spent so far ${budget.spent(rows)}")
        if dry_run:
            continue
        t0 = time.time()
        try:
            _wait(fn.spawn(f"configs/{profile}/{f}", runs, git).object_id)
            done.add(name)
        except Exception as e:  # a failed job is logged and skipped so the rest of the queue still runs
            print(f"FAILED   {name}: {e}")
        finally:
            budget.record(ledger, name, cfg.get("budget_hours", 0.3), (time.time() - t0) / 3600)
            vol.commit()
    print(f"spent this cycle: ${budget.spent(budget.load(ledger))} of ${budget.CAP_USD} cap")


@app.local_entrypoint()
def main(smoke: bool = False, dry_run: bool = False, allow_partial: bool = False):
    """--allow-partial: only with the workspace spend limit at $0 (see querycraft/budget.py)."""
    git = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"], text=True):
        git += "-dirty"
    queue.remote(smoke, dry_run, git, allow_partial)  # use `modal run --detach` so it keeps going after this terminal closes
