#!/usr/bin/env bash
# Download and unpack BIRD train + dev. Resumable (curl -C -), idempotent.
set -euo pipefail
RAW=${DATA_DIR:-data}/raw
mkdir -p "$RAW"
BASE=https://bird-bench.oss-cn-beijing.aliyuncs.com
for split in dev train; do
  zip="$RAW/$split.zip"
  if [ ! -f "$RAW/$split/.done" ]; then
    curl -L --retry 20 --retry-delay 5 -C - -o "$zip" "$BASE/$split.zip"
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).testzip()" "$zip" >/dev/null
    rm -rf "$RAW/$split"; mkdir -p "$RAW/$split"
    python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$zip" "$RAW/$split"
    find "$RAW/$split" -name "__MACOSX" -prune -exec rm -rf {} +  # macOS resource forks, not real zips
    # databases ship as a nested zip
    for inner in $(find "$RAW/$split" -name "*_databases.zip" -not -name "._*"); do
      python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$inner" "$(dirname "$inner")"
      rm "$inner"
    done
    find "$RAW/$split" -name "__MACOSX" -prune -exec rm -rf {} +
    touch "$RAW/$split/.done"
  fi
done
echo "BIRD ready in $RAW"
