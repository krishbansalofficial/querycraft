"""YAML configs with single inheritance, plus per-run provenance logging."""
import json
import os
import platform
import subprocess
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

import yaml

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", "runs"))


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path) -> dict:
    """Load YAML; `inherit: other.yaml` (relative to this file) is merged underneath."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    cfg.setdefault("name", path.stem)
    cfg["_path"] = str(path)
    return cfg


def run_dir(cfg) -> Path:
    d = RUNS_DIR / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _git_commit():
    if os.environ.get("QC_GIT_COMMIT"):  # set by the Modal launcher; the container has no .git
        return os.environ["QC_GIT_COMMIT"]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip())
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


PKGS = ["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "vllm", "datasets", "sqlglot"]


def provenance(cfg) -> dict:
    versions = {}
    for p in PKGS:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = f"{torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}"
    except Exception:
        pass
    return {
        "seed": cfg.get("seed"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg.get("_path"),
        "git_commit": _git_commit(),
        "packages": versions,
        "python": platform.python_version(),
        "gpu": gpu,
        "host": platform.node(),
    }


@contextmanager
def logged_stage(cfg, stage: str):
    """Write runs/<name>/<stage>_meta.json with provenance and wall clock, even on failure."""
    meta = provenance(cfg)
    meta["stage"] = stage
    meta["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.time()
    status = "failed"
    try:
        yield meta  # stages may add fields (e.g. metrics paths)
        status = "ok"
    finally:
        meta["status"] = status
        meta["wall_clock_s"] = round(time.time() - t0, 1)
        out = run_dir(cfg) / f"{stage}_meta.json"
        out.write_text(json.dumps(meta, indent=2, default=str))
