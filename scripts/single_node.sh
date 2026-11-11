#!/usr/bin/env bash
# Configuration export by default. --run explicitly starts the selected phase.
set -euo pipefail
phase="${1:-export}"
mode="${2:---export}"
case "$phase" in rollout|train|online) ;; *) echo 'Usage: single_node.sh rollout|train|online [--export|--run]' >&2; exit 2;; esac
case "$mode" in --export|--run) ;; *) exit 2;; esac
for setting in PYTHON_BIN CRG_PATH XTUNER_PATH LMDEPLOY_PATH LAGENT_PATH MODEL_PATH WORK_DIR XTUNER_RUN_ID DATA_PATH CYBERGYM_TASKS_DIR RL_LLM_BASE_URL RL_LLM_ADMIN_URL; do
  if [[ -z "${!setting:-}" ]]; then echo "Missing $setting" >&2; exit 2; fi
done
export PYTHONPATH="$CRG_PATH:$CRG_PATH/src:$XTUNER_PATH:$LMDEPLOY_PATH:$LAGENT_PATH${PYTHONPATH:+:$PYTHONPATH}"
export XTUNER_USE_LMDEPLOY=1 XTUNER_USE_VLLM=0 XTUNER_USE_SGLANG=0
# Published CRG !78 currently has a recipe-only selector. XTuner never reads it.
# The reviewed native-only local recipe does not require a selector.
export CYBERGYM_MTP_ROUTE=native
unset FROZEN_MTP_SOURCE FROZEN_MTP_MANIFEST FROZEN_MTP_SHA256 XT_FROZEN_MTP_SOURCE
args=(--phase "$phase")
if [[ "$mode" == --run ]]; then args+=(--run); fi
exec "$PYTHON_BIN" -m recipe.cybergym.main "${args[@]}"
