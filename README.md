# Agent RL: scripts and quick validation materials

This branch carries reviewer material, separately from the XTuner code PRs. No credentials, task data, model weights, private deployment addresses, or generated checkpoints are included.

## Single-node recipe

`scripts/single_node.sh` wraps the published CRG `recipe.cybergym.main` entry. Choose `rollout`, `train`, or `online`; configuration export is the default, and only `--run` starts training. Set every required path/endpoint in your trusted environment. Use the reviewed Python 3.12 / torch 2.9 / transformers 5.17 environment with GLM5Next present, compatible LMDeploy, and sandbox SDK for natural tasks. The wrapper does not install dependencies or supply credentials.

For a new debugging run, collect three continuous `debug_rollout_N.pt` files first, then train-only replay from that directory. Existing compatible PT may be reused. Ordinary JSONL is not a substitute. Use distinct WORK_DIR values and run IDs for phases. Export config and check actual source origins, batch units, timeouts and resources before running.

The single-node recipe uses 8 shared H200 GPUs, SP4/EP1 training, TP1/EP1 inference, 8 groups x K8, Agent concurrency64, 8192 generated tokens/request, context/pack262144, frozen native MTP (1 layer, draft3). These are recipe values, not generic XTuner defaults. Check the exported effective config; script environment and recipe defaults can override one another. Do not run on occupied GPUs.

## Controlled quick GPU regression

`quick_gpu/gpu_jobs/{driver,config,controlled_replay,training_observer,pipeline}.py` and tasks.jsonl are copied from the completed reviewed experiment without implementation changes. They exercise two real updates with an intact recorded training group, live native task retention, real HTTP/session affinity/provenance/R3, IPC synchronization, full Trainer DCP save and separate-process cold restore.

Only the environment adapter (`runtime.py`) and new `prepare.py` / `run.sh` wrapper are portability edits. They have received syntax checks, not a fresh GPU execution. The historical pass belongs to the recorded original runner and exact inputs; it must not be attributed to this new wrapper. The quick config requires the reviewed native-only CRG recipe (local revision a194e2e9); current CRG !78 retains a selector, so do not silently substitute its different entry.

Set PYTHON_BIN, REPLAY_PT, MODEL_PATH, XTUNER_PATH (assembled stack + native #2292), LMDEPLOY_PATH, LAGENT_PATH, CRG_PATH, DATA_PATH, CYBERGYM_TASKS_DIR, SPLIT_RJOB_WORK_DIR. Allocate fresh checkout/output/ports. `bash quick_gpu/run.sh --prepare` records immutable input identities. Confirm the native PT loader and per-replica packing, including nonempty DP replicas, then `bash quick_gpu/run.sh --run`. REPLAY_PT must contain a compatible complete training group; neither preparation nor the runner manufactures missing fields or rewrites labels/rewards/R3. Natural CyberGym credentials are not needed by the controlled loop, but the recipe still requires config paths.

The runner keeps dedicated MTP tensors out of the optimizer and audits all training ranks. It compares bounded main/shared parameters, full local MTP hashes and optimizer steps; it does not independently compare every optimizer moment. It does not restore live Agent coroutines or perform a third post-restore update. It deletes its temporary checkpoint after successful cold restore and retains receipts. Cleanup methods target the runner's own controller/Ray context; do not use global ray stop or kill other users' processes. This portable wrapper has no SSH/RJob launcher or detached-process supervisor: run foreground and verify owned processes/GPU release after exit. A failed run may retain its checkpoint for diagnosis.

This is engineering regression evidence, not natural task reward learning, MTP speedup, convergence or 64-GPU acceptance.
