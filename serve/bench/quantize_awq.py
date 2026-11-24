"""4 bit AWQ (W4A16) of a merged SFT model with llm-compressor. Runs in its own venv (.venv-quant) so llm-compressor's
pins never touch the evaluation stack.

    .venv-quant/bin/python serve/bench/quantize_awq.py runs/qwen1.5b_lora_r16/grpo/sft_merged runs/qwen1.5b_lora_r16/awq

Calibration: 64 val prompts with their gold SQL, up to 1,024 tokens (the split choices are made on; dev and held back
are not read). 256 x 2,048 spilled past 8 GB (32 s per AWQ grid step, about 20 h); 64 x 1,024 fits.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(src, dst, n=64, max_len=1024, tokenizer="Qwen/Qwen2.5-Coder-1.5B-Instruct"):
    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from querycraft.data import load_split
    from querycraft.prompts import PromptBuilder
    tok = AutoTokenizer.from_pretrained(tokenizer)
    pb = PromptBuilder(tok, 1280)
    rows = load_split("val")[:n]
    texts = [tok.apply_chat_template(pb.messages(r) + [{"role": "assistant", "content": r["sql"]}], tokenize=False)
             for r in rows]
    ds = Dataset.from_dict({"text": texts}).map(lambda x: tok(x["text"], truncation=True, max_length=max_len),
                                                remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")
    t0 = time.time()
    oneshot(model=model, dataset=ds, recipe=[AWQModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])],
            max_seq_length=max_len, num_calibration_samples=len(texts))
    Path(dst).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst, save_compressed=True)
    tok.save_pretrained(dst)
    meta = {"source": src, "scheme": "W4A16_ASYM", "calibration": f"{len(texts)} val prompts + gold SQL",
            "seconds": round(time.time() - t0, 1), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (Path(dst) / "querycraft_quant.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
