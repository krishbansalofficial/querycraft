"""vLLM inference (evaluation only). One engine per base model; LoRA checkpoints are served as adapters.
DoRA adapters (unsupported by vLLM LoRA) and full fine tunes are merged/loaded as plain models."""
import json
import os
import random
from pathlib import Path

# Greedy decoding needs no top-k/top-p kernel; FlashInfer's sampler would JIT compile one with nvcc,
# which a driver only machine (WSL, slim pods) does not have.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from .data import load_split
from .prompts import PromptBuilder, extract_sql


def is_dora(ckpt) -> bool:
    cfg = Path(ckpt) / "adapter_config.json"
    return cfg.exists() and json.loads(cfg.read_text()).get("use_dora", False)


def is_adapter(ckpt) -> bool:
    return ckpt is not None and (Path(ckpt) / "adapter_config.json").exists()


def merge_adapter(base_model, ckpt) -> str:
    """Merge an adapter into the bf16 base and save it next to the checkpoint (cached)."""
    out = Path(ckpt).parent.parent / "merged" / Path(ckpt).name
    if (out / "config.json").exists():
        return str(out)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, ckpt).merge_and_unload()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out)
    del model
    return str(out)


class Generator:
    def __init__(self, cfg, model_path=None):
        from vllm import LLM
        e = cfg["eval"]
        self.cfg = cfg
        self.model_path = model_path or cfg["model"]
        self.llm = LLM(model=self.model_path, tokenizer=cfg["model"], dtype="bfloat16", seed=cfg["seed"],
                       max_model_len=e["max_model_len"], gpu_memory_utilization=e["gpu_memory_utilization"],
                       enable_lora=e.get("enable_lora", True), max_lora_rank=64, max_loras=1,
                       enforce_eager=e.get("enforce_eager", False))
        self.tok = self.llm.get_tokenizer()
        self._lora_ids = {}

    def _lora(self, adapter):
        if adapter is None:
            return None
        from vllm.lora.request import LoRARequest
        i = self._lora_ids.setdefault(adapter, len(self._lora_ids) + 1)
        return LoRARequest(f"a{i}", i, adapter)

    def generate(self, prompts, adapter=None, max_tokens=None):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens or self.cfg["eval"]["max_new_tokens"])
        outs = self.llm.generate(prompts, sp, lora_request=self._lora(adapter))
        return [o.outputs[0].text for o in outs]


def pick_shots(cfg):
    k = cfg["prompt"].get("shots", 0)
    if not k:
        return []
    train = load_split("train")  # few shot examples come from train only
    return random.Random(cfg["seed"]).sample(train, k)


def predict(gen: Generator, cfg, split, adapter=None, out_path=None):
    rows = load_split(split)
    if cfg["eval"].get("limit"):
        rows = rows[: cfg["eval"]["limit"]]
    pb = PromptBuilder(gen.tok, cfg["prompt"]["schema_budget"], pick_shots(cfg))
    prompts = [pb.prompt_text(ex) for ex in rows]
    raw = gen.generate(prompts, adapter)
    preds = [{"id": ex["id"], "db_id": ex["db_id"], "db_path": ex["db_path"], "difficulty": ex["difficulty"],
              "question": ex["question"], "gold_sql": ex["sql"], "raw": r, "pred_sql": extract_sql(r),
              "prompt_tokens": pb.n_tokens(p)} for ex, r, p in zip(rows, raw, prompts)]
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("".join(json.dumps(p) + "\n" for p in preds))
    return preds
