"""Alternative: serve the same stack on Modal (vLLM + API in one L4 container). NOT RUN by this project.
`modal deploy serve/deploy/modal_serve.py` would bill an L4 (~$0.80/h at modal.com/pricing, 2026-11-23) while a
container is warm; scaledown_window keeps idle cost near zero. Spend limit at $0 means it stops instead of charging."""
import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2]
image = (modal.Image.from_registry("vllm/vllm-openai:v0.32.0", add_python=None)
         .pip_install("fastapi", "uvicorn", "prometheus-client", "opentelemetry-sdk", "httpx", "sentence-transformers",
                      "faiss-cpu", "lightgbm", "scikit-learn", "joblib", "sqlglot", "pyyaml")
         .add_local_dir(REPO, "/app", ignore=[".venv", ".git", "data/raw/train", "runs/*/eval", "runs_*"]))
app = modal.App("querycraft-serve", image=image)


@app.function(gpu="L4", timeout=24 * 3600, scaledown_window=300)
@modal.concurrent(max_inputs=32)
@modal.web_server(8000, startup_timeout=600)
def serve():
    subprocess.Popen("vllm serve Qwen/Qwen2.5-Coder-1.5B-Instruct --enable-lora --max-lora-rank 64 "
                     "--lora-modules querycraft=/app/runs/qwen1.5b_lora_r16/ckpt/checkpoint-420 --port 8001 "
                     "--max-model-len 4096 --gpu-memory-utilization 0.85", shell=True)
    subprocess.Popen("cd /app && uvicorn serve.app:app --host 0.0.0.0 --port 8000", shell=True)
