#!/usr/bin/env bash
# One rented session: env check, data, tests, every baseline + trained run for $PROFILE back to back,
# report, then stop the pod. A failed run is logged and skipped so the rest of the paid session is not lost.
#
# RunPod: attach a network volume at /workspace (it persists after the pod stops; that is the synced folder),
#   cd /workspace && git clone <repo> querycraft && cd querycraft && nohup bash scripts/runpod_session.sh &
# Local rehearsal (0.5B, 20 examples, no shutdown):   make smoke
set -uo pipefail
cd "$(dirname "$0")/.."
PROFILE=${PROFILE:-rented}
SHUTDOWN=${SHUTDOWN:-1}
if [ -d /workspace ] && [ "$PROFILE" = rented ]; then
  export DATA_DIR=${DATA_DIR:-/workspace/data} HF_HOME=${HF_HOME:-/workspace/hf}
  export RUNS_DIR=${RUNS_DIR:-/workspace/querycraft_sync/runs}
fi
export RUNS_DIR=${RUNS_DIR:-runs} PYTHONPATH=$PWD
SYNC=$(dirname "$RUNS_DIR")/session_logs
mkdir -p "$RUNS_DIR" "$SYNC"
LOG="$SYNC/${PROFILE}_$(date +%Y%m%d_%H%M%S).log"
PY=.venv/bin/python
FAILED=()

finish() {
  echo "== session done $(date). failed: ${FAILED[*]:-none}" | tee -a "$LOG"
  cp -f README.md "$SYNC/README.generated.md" 2>/dev/null || true
  if [ "$SHUTDOWN" = 1 ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "stopping pod $RUNPOD_POD_ID" | tee -a "$LOG"
    runpodctl stop pod "$RUNPOD_POD_ID"
  fi
}
trap finish EXIT

step() { echo "== $* $(date +%H:%M:%S)" | tee -a "$LOG"; "$@" >>"$LOG" 2>&1 || { FAILED+=("$*"); echo "!! FAILED: $*" | tee -a "$LOG"; return 1; }; }

[ -x $PY ] || step bash scripts/setup_env.sh
step $PY scripts/verify_env.py
step make data PY=$PY
step $PY -m pytest -q tests
for c in $(ls configs/$PROFILE/0*.yaml 2>/dev/null); do step $PY -m querycraft.run_eval "$c"; done
for c in $(ls configs/$PROFILE/*.yaml | grep -v "/0[^/]*$"); do
  step $PY -m querycraft.train "$c" && step $PY -m querycraft.run_eval "$c"
done
step $PY -m querycraft.report
[ ${#FAILED[@]} -eq 0 ]
