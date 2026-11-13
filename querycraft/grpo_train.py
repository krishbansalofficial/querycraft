"""GRPO step 3/4: reinforcement learning with an execution reward on top of the SFT LoRA (docs/plans/grpo.md).

    python -m querycraft.grpo_train configs/local/11_qwen1.5b_lora_r16.yaml binary|shaped [max_steps]

The SFT adapter is merged into the base once (runs/<sft>/grpo/sft_merged); a fresh LoRA r16 is trained on top.
Arms B (binary) and C (shaped) share everything but the reward. Arm C also logs the binary reward with weight 0,
so the gap between shaped reward and real accuracy (reward hacking) is visible in the same log.
"""
import json
import sys

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .config import load_config, logged_stage, run_dir
from .data import load_split
from .prompts import PromptBuilder
from .reward import binary_reward, shaped_reward
from .train import PruneCheckpoints

# fixed before seeing any dev number. The plan's 250 steps measured 115 s/step on the laptop (8 h per arm), over the
# 6 h guardrail, so per the plan: 1,000 prompts = 125 steps of 8 prompts, one pass, checkpoints every 25.
STEPS, SAVE_EVERY = 125, 25


def merged_sft(cfg):
    from peft import PeftModel
    d = run_dir(cfg) / "grpo" / "sft_merged"
    if not (d / "config.json").exists():
        tag = json.loads((run_dir(cfg) / "results.json").read_text())["selected_checkpoint"]
        base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16)
        PeftModel.from_pretrained(base, run_dir(cfg) / "ckpt" / tag).merge_and_unload().save_pretrained(d)
        AutoTokenizer.from_pretrained(cfg["model"]).save_pretrained(d)
    return d


def main(cfg_path, arm, max_steps=STEPS):
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    sft = load_config(cfg_path)
    cfg = sft | {"name": f"{sft['name'].split('_')[0]}_grpo_{arm}", "grpo": {"arm": arm, "from": sft["name"]}}
    set_seed(cfg["seed"])
    src = merged_sft(sft)
    tok = AutoTokenizer.from_pretrained(src)
    pb = PromptBuilder(tok, cfg["prompt"]["schema_budget"])
    by_id = {r["id"]: r for r in load_split("train")}  # train only: the reward never sees val/dev/held back
    keep = [json.loads(l) for l in (run_dir(sft) / "grpo" / "prompts.jsonl").read_text().splitlines()]
    ds = Dataset.from_list([{"prompt": pb.messages(by_id[k["id"]]), "db_path": k["db_path"], "gold_sql": k["gold_sql"]}
                            for k in keep])
    rewards, weights = ([binary_reward], [1.0]) if arm == "binary" else ([shaped_reward, binary_reward], [1.0, 0.0])
    out = run_dir(cfg) / "ckpt"
    with logged_stage(cfg, "train") as meta:
        args = GRPOConfig(
            output_dir=str(out), run_name=cfg["name"], seed=cfg["seed"], data_seed=cfg["seed"],
            max_steps=max_steps, learning_rate=1e-5, lr_scheduler_type="constant", beta=0.0,
            num_generations=8, per_device_train_batch_size=8, gradient_accumulation_steps=8,  # 8 prompts x 8 samples
            # generating all 64 at once spilled past 8 GB (418 s/step); vLLM colocate needs a second 3 GB copy
            # that does not fit, so generate one prompt's group of 8 per micro batch instead
            steps_per_generation=1,
            temperature=1.0, max_completion_length=256, reward_weights=weights,
            bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=1, save_strategy="steps", save_steps=SAVE_EVERY, report_to="none",
        )
        trainer = GRPOTrainer(
            model=AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, attn_implementation="sdpa"),
            reward_funcs=rewards, args=args, train_dataset=ds, processing_class=tok,
            peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear",
                                   task_type="CAUSAL_LM"),
            callbacks=[PruneCheckpoints(range(SAVE_EVERY, STEPS + 1, SAVE_EVERY))])
        resume = any(out.glob("checkpoint-*"))
        trainer.train(resume_from_checkpoint=resume or None)
        hist = trainer.state.log_history
        (run_dir(cfg) / "train_log.json").write_text(json.dumps(hist, indent=1))
        steps = [h for h in hist if "step_time" in h or "reward" in h]
        meta |= {"n_prompts": len(ds), "resumed": resume, "base": str(src),
                 "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "n_logged_steps": len(steps)}


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:]))
