#!/usr/bin/env bash
# Bring Kaggle notebook outputs back into runs/: download each notebook's /kaggle/working/out/*.tar.gz (Output tab,
# or `kaggle kernels output <user>/<notebook> -p downloads/`), then:   bash scripts/ingest_kaggle.sh downloads/*.tar.gz
# notebook_a / notebook_b unpack into runs/ (+ configs/kaggle_*.yaml); qwen7b_dora_awq into runs/qwen7b_dora_best/awq.
set -euo pipefail
cd "$(dirname "$0")/.."
for f in "$@"; do
  case "$(basename "$f")" in
    qwen7b_dora_awq*) mkdir -p runs/qwen7b_dora_best && tar xzf "$f" -C runs/qwen7b_dora_best && \
                      rm -rf runs/qwen7b_dora_best/awq && mv runs/qwen7b_dora_best/qwen7b_dora_awq runs/qwen7b_dora_best/awq ;;
    *) tar xzf "$f" -C . ;;
  esac
  echo "ingested $f"
done
# Kaggle paths inside the copied configs point at /kaggle/...; the per question files are what the report reads.
.venv/bin/python -m querycraft.report >/dev/null && echo "report rebuilt (make audit next)"
