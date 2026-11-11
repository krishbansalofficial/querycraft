"""Step 1: prove the stack works on this GPU (torch CUDA, bitsandbytes 4-bit matmul, vLLM import)."""
import json
import sys

import torch

out = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
assert torch.cuda.is_available(), "torch does not see the GPU"
out["gpu"] = torch.cuda.get_device_name(0)
out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
out["bf16"] = torch.cuda.is_bf16_supported()
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
out["matmul_ok"] = bool(torch.isfinite(x @ x).all())

import bitsandbytes as bnb

lin = bnb.nn.Linear4bit(256, 256, compute_dtype=torch.bfloat16, quant_type="nf4").cuda()
y = lin(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16))
out["bitsandbytes"] = bnb.__version__
out["bnb_nf4_ok"] = bool(torch.isfinite(y).all())

import vllm

out["vllm"] = vllm.__version__
print(json.dumps(out, indent=2))
sys.exit(0 if out["matmul_ok"] and out["bnb_nf4_ok"] else 1)
