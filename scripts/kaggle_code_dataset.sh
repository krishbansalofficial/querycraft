#!/usr/bin/env bash
# Upload (or version) the private Kaggle dataset krishbansalofficial/querycraft-code that the notebooks read:
# the package, configs, and the small run outputs the 7B jobs need (retriever tables, agent options). No model weights.
set -euo pipefail
cd "$(dirname "$0")/.."
K=${KAGGLE:-.venv/bin/kaggle}
S=$(mktemp -d)/code
mkdir -p "$S/runs/table_retriever" "$S/runs/agent_qwen1.5b"
cp -r querycraft configs "$S/"
find "$S" -name __pycache__ -prune -exec rm -rf {} +
cp -r runs/table_retriever/ranked runs/table_retriever/eval runs/table_retriever/sweep.json "$S/runs/table_retriever/"
cp runs/agent_qwen1.5b/results.json "$S/runs/agent_qwen1.5b/"
echo '{"title": "querycraft code", "id": "krishbansalofficial/querycraft-code", "licenses": [{"name": "other"}]}' > "$S/dataset-metadata.json"
if $K datasets status krishbansalofficial/querycraft-code >/dev/null 2>&1; then
  $K datasets version -p "$S" -m "code $(git rev-parse --short HEAD)" --dir-mode zip
else
  $K datasets create -p "$S" --dir-mode zip
fi
