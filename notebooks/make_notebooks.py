"""Write the Kaggle notebooks for the 7B jobs that do not fit the 8 GB laptop (docs/plans/v2_compute.md).

    python notebooks/make_notebooks.py     # writes notebooks/kaggle_{a,b,c}_*.ipynb

Each notebook: Accelerator "GPU T4 x2", Internet on, and three private datasets attached as inputs:
krishbansalofficial/querycraft-code (scripts/kaggle_code_dataset.sh), querycraft-bird-eval, querycraft-7b-dora.
Outputs land in /kaggle/working/out/*.tar.gz; bring them back with scripts/ingest_kaggle.sh.
T4 has no bf16, so 7B runs in fp16 split over both GPUs (tensor parallel 2).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

SETUP = r'''
import glob, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
def sh(cmd):
    print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
IN = Path("/kaggle/input")
def find(slug):
    hits = [p for p in IN.glob("*") if slug in p.name] + [p for p in IN.glob("*/*/*") if slug in str(p)]
    assert hits, f"attach the dataset {slug} (Add Input)"; return hits[0]
W, T = Path("/kaggle/working"), Path("/kaggle/tmp")
QC = T / "qc"; OUT = W / "out"  # code + data in tmp (not saved); only out/ is kept
OUT.mkdir(exist_ok=True); T.mkdir(exist_ok=True)
def unpack(src, dst):
    """Kaggle may keep uploaded folders as .zip files: extract those, copy the rest."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in Path(src).iterdir():
        if p.suffix == ".zip":
            zipfile.ZipFile(p).extractall(dst)
        elif p.is_dir():
            shutil.copytree(p, dst / p.name, dirs_exist_ok=True)
        else:
            shutil.copy(p, dst / p.name)
unpack(find("querycraft-code"), QC)
unpack(find("querycraft-bird-eval"), QC)            # data/raw/... and data/processed/...
unpack(find("querycraft-7b-dora"), T / "runs/qwen7b_dora_best/ckpt")
os.chdir(QC); os.environ["PYTHONPATH"] = str(QC); sys.path.insert(0, str(QC))
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
(QC / "runs").mkdir(exist_ok=True)
print(sorted(p.name for p in QC.iterdir()))
'''

INSTALL_VLLM = r'''
# the same pinned stack as the laptop and Modal (vLLM 0.32, torch 2.13 cu129)
sh("pip install -q uv")
sh("uv pip install --system -q --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu129 "
   "'torch==2.13.0+cu129' 'vllm @ https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl' "
   "'transformers>=5.16.1,<5.20.0' peft sqlglot pyyaml opentelemetry-api httpx")
sh("uv pip uninstall --system torchcodec || true")
'''

MERGE = r'''
# DoRA is not servable by vLLM: merge it into the 7B base on CPU (bf16 weights, ~15 GB in /kaggle/tmp)
from querycraft.infer import merge_adapter
t0 = time.time()
MERGED = merge_adapter("Qwen/Qwen2.5-Coder-7B-Instruct", str(T / "runs/qwen7b_dora_best/ckpt/checkpoint-560"))
print(MERGED, f"{time.time() - t0:.0f} s")
'''

CFG_7B = r'''
import yaml
def cfg_7b(name, linking=None, extra=None):
    c = {"inherit": str(QC / "configs/rented/01_qwen7b_zeroshot.yaml"), "name": name, "model": MERGED,
         "eval": {"dtype": "float16", "tensor_parallel_size": 2, "gpu_memory_utilization": 0.85,
                  "tokenizer": "Qwen/Qwen2.5-Coder-7B-Instruct", "forgetting": False,
                  "splits": ["val", "dev", "heldback"]}}
    if linking:
        c["prompt"] = {"linking": "retriever", "link_from": "runs/table_retriever/eval/{split}_tables.jsonl"}
    c.update(extra or {})
    p = QC / f"configs/kaggle_{name}.yaml"; p.write_text(yaml.safe_dump(c)); return str(p)
'''

SMOKE = r'''
# 5 minute smoke test: does vLLM 0.32 run on T4 (compute capability 7.5) in fp16 with tensor parallel 2?
# If this cell fails, stop here: the fallback is the Modal A100 after Dec 1 (docs/plans/v2_compute.md).
sh(f"python - <<'PY'\nimport sys; sys.path.insert(0, '{QC}')\nfrom querycraft.config import load_config\n"
   f"from querycraft.infer import Generator\nfrom querycraft.prompts import PromptBuilder\nfrom querycraft.data import load_split\n"
   f"c = load_config('{cfg_7b('smoke')}'); g = Generator(c, c['model']); pb = PromptBuilder(g.tok, 1280)\n"
   f"rows = load_split('dev')[:8]; import time; t = time.time(); out = g.generate([pb.prompt_text(r) for r in rows])\n"
   f"print(f'{{time.time() - t:.1f}} s for 8 prompts'); print(out[:3])\nPY")
'''

PACK = r'''
def pack(name, paths):
    dst = OUT / f"{name}.tar.gz"
    sh(f"cd {QC} && tar czf {dst} " + " ".join(paths)); print(dst, dst.stat().st_size // 2**20, "MB")
'''


def nb(title, cells):
    md = {"cell_type": "markdown", "metadata": {}, "source": [title]}
    code = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [c.strip() + "\n"]}
            for c in cells]
    return {"cells": [md] + code, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                               "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


A = nb("""# Querycraft v2 notebook A: 7B DoRA single pass on 2x T4 (fp16)

Settings: **GPU T4 x2**, **Internet on**. Inputs: `querycraft-code`, `querycraft-bird-eval`, `querycraft-7b-dora`.
Runs: (1) smoke test, (2) the 7B DoRA model single pass with the normal pruned schema, (3) single pass over the trained
retriever's tables. Both on val, dev and held back. These are the paired baselines for the 7B agent (notebook B),
measured on the same hardware and precision. Estimated 1.5 GPU hours. Output: `/kaggle/working/out/notebook_a.tar.gz`.
""", [SETUP, INSTALL_VLLM, MERGE, CFG_7B, SMOKE, r'''
for name, link in (("qwen7b_dora_t4_single", False), ("qwen7b_dora_t4_retrieved", True)):
    t0 = time.time()
    sh(f"python -m querycraft.run_eval {cfg_7b(name, link)}")
    print(name, f"{(time.time() - t0) / 3600:.2f} h")
''', PACK, r'''pack("notebook_a", ["runs/qwen7b_dora_t4_single", "runs/qwen7b_dora_t4_retrieved", "configs/kaggle_*.yaml"])'''])

B = nb("""# Querycraft v2 notebook B: 7B DoRA agent loop on 2x T4 (fp16)

Settings: **GPU T4 x2**, **Internet on**. Same three inputs as notebook A. Runs the agent (querycraft/agent.py) with the
options the 1.5B tuning chose on val (no second tuning grid on Kaggle hours), on val, dev and held back, plus 50 dev
questions timed one at a time. Estimated 5 GPU hours; results are written per split, so a cut session keeps finished
splits. Output: `/kaggle/working/out/notebook_b.tar.gz`.
""", [SETUP, INSTALL_VLLM, MERGE, CFG_7B, r'''
opts = json.loads((QC / "runs/agent_qwen1.5b/results.json").read_text())["options"]
agent_cfg = {"inherit": str(QC / "configs/agent/agent_1.5b.yaml"), "name": "agent_qwen7b",
             "sql_config": cfg_7b("qwen7b_dora_t4_retrieved_sql", True),
             "defaults": {k: opts[k] for k in ("repair_on_empty", "widen_on_error")}, "grid": None}
p = QC / "configs/kaggle_agent_qwen7b.yaml"; p.write_text(yaml.safe_dump(agent_cfg))
t0 = time.time()
sh(f"python -m querycraft.agent {p}")
print(f"{(time.time() - t0) / 3600:.2f} h")
''', PACK, r'''pack("notebook_b", ["runs/agent_qwen7b", "configs/kaggle_*.yaml"])'''])

C = nb("""# Querycraft v2 notebook C: 4 bit AWQ (W4A16) of the merged 7B DoRA model

Settings: **GPU T4 x2** (or Colab T4 as a backup), **Internet on**. Same three inputs. Quantizes with llm-compressor's
AWQ recipe, calibrated on 64 val prompts (up to 1,024 tokens) with their gold SQL, as for the 1.5B model (val is the split choices are made on; dev and held
back are never touched). Output: `/kaggle/working/out/qwen7b_dora_awq.tar.gz` (~5.5 GB), which the laptop benchmark
loads with vLLM (`make bench`). Estimated 1 GPU hour.
""", [SETUP, r'''
sh("pip install -q uv && uv pip install --system -q llmcompressor 'transformers>=5.16.1,<5.20.0' peft sqlglot pyyaml")
''', MERGE, r'''
import torch
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer
from querycraft.data import load_split
from querycraft.prompts import PromptBuilder
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
pb = PromptBuilder(tok, 1280)
rows = load_split("val")[:64]
texts = [tok.apply_chat_template(pb.messages(r) + [{"role": "assistant", "content": r["sql"]}], tokenize=False) for r in rows]
ds = Dataset.from_dict({"text": texts}).map(lambda x: tok(x["text"], truncation=True, max_length=1024), remove_columns=["text"])
model = AutoModelForCausalLM.from_pretrained(MERGED, dtype=torch.float16, device_map="auto")
recipe = [AWQModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])]
t0 = time.time()
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=1024, num_calibration_samples=len(texts))
dst = W / "qwen7b_dora_awq"
model.save_pretrained(dst, save_compressed=True); tok.save_pretrained(dst)
json.dump({"calibration": "64 val prompts + gold SQL, 1024 tokens", "scheme": "W4A16_ASYM", "seconds": round(time.time() - t0)},
          open(dst / "querycraft_quant.json", "w"))
sh(f"cd {W} && tar czf {OUT}/qwen7b_dora_awq.tar.gz qwen7b_dora_awq && rm -rf qwen7b_dora_awq")
'''])

for name, book in (("kaggle_a_7b_single_pass", A), ("kaggle_b_7b_agent", B), ("kaggle_c_7b_awq", C)):
    (HERE / f"{name}.ipynb").write_text(json.dumps(book, indent=1))
    print("wrote", HERE / f"{name}.ipynb")
