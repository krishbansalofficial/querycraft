"""SFT for text to SQL: qlora | lora | dora | full. Resumes from the latest checkpoint automatically.

    python -m querycraft.train configs/local/qwen1.5b_qlora_r16.yaml
"""
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

from .config import load_config, logged_stage, run_dir
from .data import load_split
from .prompts import PromptBuilder


def build_dataset(cfg, tok):
    t = cfg["train"]
    rows = load_split("train")
    if t.get("n_train"):
        rows = rows[: t["n_train"]]
    pb = PromptBuilder(tok, cfg["prompt"]["schema_budget"])
    recs, dropped = [], 0
    for ex in rows:
        prompt = pb.messages(ex)
        completion = [{"role": "assistant", "content": ex["sql"]}]
        n = len(tok.apply_chat_template(prompt + completion, tokenize=True))
        if n > t["max_len"]:  # never train on truncated targets
            dropped += 1
            continue
        recs.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(recs), {"n_examples": len(recs), "dropped_too_long": dropped}


def load_model(cfg):
    t = cfg["train"]
    method = t["method"]
    kw = {"attn_implementation": "sdpa"}
    if method == "qlora" or (method == "dora" and t.get("quantize", True)):
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kw["dtype"] = torch.bfloat16
    elif method == "full":
        kw["dtype"] = torch.float32  # fp32 master weights; bf16 autocast. bf16 weights would swallow 1e-5 updates.
    else:
        kw["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **kw)
    peft_config = None
    if method != "full":
        from peft import LoraConfig, prepare_model_for_kbit_training
        if "quantization_config" in kw:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        peft_config = LoraConfig(
            r=t["rank"], lora_alpha=t.get("alpha", 2 * t["rank"]), lora_dropout=t.get("dropout", 0.05),
            target_modules="all-linear", use_dora=(method == "dora"), task_type="CAUSAL_LM")
    return model, peft_config


def resolve_rank(t):
    """rank: best -> the sweep run with the highest validation EX (never dev)."""
    if t["rank"] != "best":
        return t
    from .config import RUNS_DIR
    scores = {}
    for name in t["rank_from"]:
        r = json.loads((RUNS_DIR / name / "results.json").read_text())
        scores[name] = (max(r["val_ex_by_checkpoint"].values()), json.loads((RUNS_DIR / name / "train_meta.json").read_text())["config"]["train"]["rank"])
    best = max(scores.values())  # ties -> larger rank
    t["rank"], t["alpha"] = best[1], 2 * best[1]
    t["rank_selection"] = {k: v[0] for k, v in scores.items()}
    return t


def main(cfg_path):
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(cfg_path)
    t = resolve_rank(cfg["train"])
    set_seed(cfg["seed"])
    out = run_dir(cfg)
    ckpt_dir = out / "ckpt"
    with logged_stage(cfg, "train") as meta:
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        ds, ds_stats = build_dataset(cfg, tok)
        meta["data"] = ds_stats
        model, peft_config = load_model(cfg)
        use_wandb = bool(os.environ.get("WANDB_API_KEY"))
        args = SFTConfig(
            output_dir=str(ckpt_dir), run_name=cfg["name"], seed=cfg["seed"], data_seed=cfg["seed"],
            max_steps=t["max_steps"], per_device_train_batch_size=t["per_device_batch"],
            gradient_accumulation_steps=t["grad_accum"], learning_rate=float(t["lr"]),
            lr_scheduler_type="cosine", warmup_steps=int(t.get("warmup_ratio", 0.03) * t["max_steps"]),
            weight_decay=t.get("weight_decay", 0.0), max_grad_norm=1.0, optim=t.get("optim", "adamw_torch"),
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            max_length=t["max_len"], completion_only_loss=True, packing=False,
            logging_steps=10, save_strategy="steps", save_steps=t["save_steps"], save_total_limit=None,
            save_only_model=False, report_to="wandb" if use_wandb else "none", dataloader_num_workers=2,
        )
        trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_config)
        if peft_config is not None:
            trainable, total = trainer.model.get_nb_trainable_parameters()
            meta["trainable_params"], meta["total_params"] = trainable, total
        resume = any(ckpt_dir.glob("checkpoint-*"))
        meta["resumed"] = resume
        trainer.train(resume_from_checkpoint=resume or None)
        if not (ckpt_dir / f"checkpoint-{t['max_steps']}").exists():
            trainer.save_model(str(ckpt_dir / f"checkpoint-{t['max_steps']}"))
        hist = trainer.state.log_history
        (out / "train_log.json").write_text(json.dumps(hist, indent=1))
        meta["final_train_loss"] = next((h["train_loss"] for h in reversed(hist) if "train_loss" in h), None)
        meta["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        meta["tokens_seen"] = int(trainer.state.num_input_tokens_seen or 0)


if __name__ == "__main__":
    main(sys.argv[1])
