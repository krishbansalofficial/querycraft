#!/usr/bin/env bash
# Local (WSL2, RTX 4070, driver 560 = CUDA 12.6) environment.
# vLLM >=0.30 ships CUDA 13 by default; we pin its cu129 build + torch cu129.
# CUDA 12.x minor version compatibility lets cu129 binaries run on a 12.6 driver
# (all kernels ship sm_89 SASS, so no driver PTX JIT is needed).
set -euo pipefail
# Triton JIT builds a C helper at runtime and needs Python.h
python3 -c "import sysconfig,os,sys; sys.exit(not os.path.exists(sysconfig.get_paths()['include'] + '/Python.h'))" \
  || (apt-get update -qq && apt-get install -y -qq python3-dev)
VLLM=0.32.0
TORCH=2.13.0
uv venv --clear --python 3.12 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  "torch==$TORCH+cu129" \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v$VLLM/vllm-$VLLM+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  -r requirements.txt
# torchcodec (audio/video only) is pulled in as a cu13 build that raises OSError on import under cu129;
# vLLM treats it as optional, so remove it. We only do text inference.
uv pip uninstall --python .venv/bin/python torchcodec
