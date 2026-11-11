#!/usr/bin/env bash
# This portable wrapper has not itself been GPU-qualified. See README.
set -euo pipefail
for setting in PYTHON_BIN REPLAY_PT SPLIT_RJOB_WORK_DIR MODEL_PATH XTUNER_PATH LMDEPLOY_PATH LAGENT_PATH CRG_PATH DATA_PATH CYBERGYM_TASKS_DIR; do
 if [[ -z "${!setting:-}" ]]; then echo "Missing $setting" >&2; exit 2; fi
done
case "${1:-}" in --prepare) exec "$PYTHON_BIN" "$(dirname "$0")/prepare.py" ;; --run) ;; *) echo 'Use --prepare or --run'; exit 2;; esac
# Same physical host cooperating experiment lock; never stop another process.
exec 9>/tmp/xt_rl_agent_gpu_experiment.lock
flock -n 9 || { echo 'Another cooperating GPU experiment holds the lock' >&2; exit 1; }
[[ ! -e "$SPLIT_RJOB_WORK_DIR" ]] || { echo 'Use a new output directory' >&2; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo 'GPU processes already present' >&2; exit 1; }
while IFS=, read -r model used; do
 [[ "$model" == *H200* && "${used// /}" == 0 ]] || { echo 'Expected idle H200 GPUs' >&2; exit 1; }
done < <(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits)
exec "$PYTHON_BIN" "$(dirname "$0")/gpu_jobs/pipeline.py"
