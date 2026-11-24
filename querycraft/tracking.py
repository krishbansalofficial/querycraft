"""MLflow tracking (v2 part 5): every run in runs/ as an MLflow run, every servable model as a registered version.

Local only: a SQLite file backend (mlflow/mlflow.db) with artifacts in mlflow/artifacts; no server, nothing leaves
the machine. W&B stays as before (train.py logs to it only when WANDB_API_KEY is set). The source of truth is still
runs/*/results.json and *_meta.json; this mirrors them so runs can be compared in the MLflow UI:

    python -m querycraft.tracking            # idempotent: one MLflow run per runs/<name>, tagged with git commit
    mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
"""
import json
from pathlib import Path

from .config import RUNS_DIR

URI = "sqlite:///mlflow/mlflow.db"
EXPERIMENT = "querycraft"
MODELS = {  # registered name -> (run name, servable artifact)
    "querycraft-sql-1.5b-lora": ("qwen1.5b_lora_r16", "ckpt/checkpoint-420"),
    "querycraft-table-retriever": ("table_retriever", "bi"),
    "querycraft-table-reranker": ("table_retriever", "cross"),
    "querycraft-abstain": ("abstain_lgbm", "model.joblib"),
}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= _flat(v, key + ".")
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def metrics_of(res):
    """Headline numbers only (EX, set metrics, calibration); per question files stay in runs/."""
    keep = {}
    for split in ("val", "dev", "heldback"):
        s = res.get(split)
        if isinstance(s, dict):
            for k in ("ex", "ex_bird", "recall", "precision", "all_gold_kept", "mean_predicted", "mean_steps",
                      "mean_tokens", "share_repaired"):
                if isinstance(s.get(k), (int, float)):
                    keep[f"{split}.{k}"] = float(s[k])
            for side in ("learned", "baseline"):
                for k in ("brier", "ece", "auroc"):
                    if isinstance((s.get(side) or {}).get(k), (int, float)):
                        keep[f"{split}.{side}.{k}"] = float(s[side][k])
    return keep


def sync():
    import mlflow
    from mlflow.tracking import MlflowClient
    Path("mlflow/artifacts").mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(URI)
    exp = mlflow.get_experiment_by_name(EXPERIMENT) or mlflow.get_experiment(
        mlflow.create_experiment(EXPERIMENT, artifact_location=str(Path("mlflow/artifacts").resolve().as_uri())))
    client = MlflowClient()
    existing = {r.info.run_name: r.info.run_id for r in client.search_runs([exp.experiment_id], max_results=10_000)}
    run_ids = {}
    for d in sorted(p for p in RUNS_DIR.iterdir() if (p / "results.json").exists()):
        res = json.loads((d / "results.json").read_text())
        metas = {m.stem.replace("_meta", ""): json.loads(m.read_text()) for m in d.glob("*_meta.json")}
        meta = metas.get("eval") or metas.get("train") or next(iter(metas.values()), {})
        if d.name in existing:
            run_ids[d.name] = existing[d.name]
            continue
        with mlflow.start_run(experiment_id=exp.experiment_id, run_name=d.name) as run:
            mlflow.set_tags({"git_commit": meta.get("git_commit", "unknown"), "gpu": str(meta.get("gpu")),
                             "config_path": str(meta.get("config_path")), "stages": ",".join(sorted(metas))})
            params = {k: v for k, v in _flat_params(meta.get("config", {})).items()}
            if params:
                mlflow.log_params(dict(list(params.items())[:100]))
            m = metrics_of(res)
            if m:
                mlflow.log_metrics(m)
            for stage, mt in metas.items():
                if mt.get("wall_clock_s"):
                    mlflow.log_metric(f"{stage}.hours", mt["wall_clock_s"] / 3600)
            run_ids[d.name] = run.info.run_id
    for name, (run_name, artifact) in MODELS.items():
        src = RUNS_DIR / run_name / artifact
        if run_name not in run_ids or not src.exists():
            continue
        try:
            client.create_registered_model(name)
        except Exception:
            pass  # already registered
        versions = client.search_model_versions(f"name='{name}'")
        if not any(v.run_id == run_ids[run_name] for v in versions):
            client.create_model_version(name, source=str(src.resolve()), run_id=run_ids[run_name])
    print(f"mlflow: {len(run_ids)} runs, models: {[n for n in MODELS if MODELS[n][0] in run_ids]}")


def _flat_params(cfg, prefix=""):
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= _flat_params(v, key + ".")
        else:
            out[key] = str(v)[:250]
    return out


if __name__ == "__main__":
    sync()
