# GLM-5.3-Flash 本地 profile

当前训练路径写入 `xtuner.L1` 训练阶段、`xtuner.L2` 层与 loss、`xtuner.L3` 子模块和 host sync 范围。`profile/` 中的工具在本地读取原始显存快照与 Chrome trace，输出单 rank 和跨 rank 报告；采集与分析可以分开执行。

## 采集 25B 短程训练

在仓库根目录运行。`GLM-5.3-Flash-25B` 是五层主栈加一层 MTP 的 24.38B 参数裁剪模型。`PROFILE_STEP` 使用从零开始的索引，`1` 对应训练第 2 步。短程验证同时采两类数据；正式测速宜分别采集，避免显存记录影响时间结果。

```bash
env GPU_LOCK_DEVICES=0,1,2,3,4,5,6,7 \
  CONDA_ENV=/mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2 \
  WORK_DIR=.scratch/glm53_perf_25b TOTAL_STEP=2 \
  PROFILE_TIME=1 PROFILE_MEMORY=1 PROFILE_STEP=1 PROFILE_RANKS=all \
  PACK_MAX_LENGTH=4096 SAMPLE_MAX_LENGTH=2048 GLOBAL_BATCH_SIZE=8 \
  ~/github/xtuner/zdev/gpu_lock.sh bash sft_glm53_tiny.sh
```

`sft_glm53_tiny.sh` 默认使用单机 8 卡、EP4、SP1、编译模型和 `flash_mla_cudnn` DSA。训练会话写在 `.scratch/glm53_perf_25b/<会话>/`。

## 在本地分析

以下命令只读取采集文件，不需要 GPU。完整的字段和分析方法见 [时间分析指南](guide_profile_time.md)及[显存分析指南](guide_profile_mem.md)。

```bash
source /mnt/shared-storage-user/zhaopenghao/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/shared-storage-user/llmrazor-share/comm_env/pt29_glm2
RUN_DIR=.scratch/glm53_perf_25b/<会话>
ANALYSIS_DIR=.scratch/glm53_perf_25b/analysis
mkdir -p "$ANALYSIS_DIR"
python profile/trace_extract.py "$RUN_DIR/profiling_time/step-1" "$ANALYSIS_DIR/parquet" --workers 2
python profile/analyze_step.py "$ANALYSIS_DIR/parquet" --rank 0 --out "$ANALYSIS_DIR/time_rank0.md"
python profile/analyze_ranks.py "$ANALYSIS_DIR/parquet" --out "$ANALYSIS_DIR/time_ranks.md" --workers 2
python profile/mem_snapshot.py "$RUN_DIR/profiling_memory/step-1" --rank 0 \
  --out "$ANALYSIS_DIR/memory_rank0.md" --png "$ANALYSIS_DIR/memory_rank0.png"
python profile/mem_ranks.py "$RUN_DIR/profiling_memory/step-1" \
  --logs "$RUN_DIR/logs" --out "$ANALYSIS_DIR/memory_ranks.md"
```

先检查 `trace_extract.py` 对每个 rank 报告的 `inside an L2 range` 大于零，再解读逐层耗时。分类规则在 `profile/profile_config.toml` 和 `profile/mem_config.toml`；GLM-5.3 的 KDA 与 SparseMLA kernel 均归入 attention。旧 trace 若未采集 L1/L2 范围，只能按 kernel 和通信事件分析，不能恢复逐层时间。

## 本地验证结果（2026-10-13）

| 会话 | 结果 |
|---|---|
| `.scratch/glm53_perf_25b/20261013131928` | 8 卡完成 2/2 步；第 2 步写出 8 份显存快照和 8 份时间 trace。rank 0 的 14,186 个 GPU 事件中有 11,293 个落入 L2。显存 active/reserved 峰值为 71.22/91.61 GiB，无分配器全量释放。 |
| `.scratch/glm53_perf_25b/20261013132616` | 8 卡完成 4/4 步；第 4 步写出 8 份时间 trace。rank 0 的 8,271 个 GPU 事件中有 5,398 个落入 L2。 |

`trace_extract.py`、`analyze_step.py`、`analyze_ranks.py`、`mem_snapshot.py` 和 `mem_ranks.py` 均在本地成功读取真实 25B 采样，报告位于 `.scratch/glm53_perf_25b/analysis/`。原始 trace 包含主栈 1–5 层及 MTP 层的 forward、recompute、backward 范围，并包含 L1 训练阶段与 L3 attention、MLP、gate 范围。

本次时间 trace 仍有约 5–6 秒的主机空档，且被采样步耗时明显高于相邻未采样步；这些报告验证打点和分析流程，不代表稳态吞吐。进一步做速度结论时，应先检查报告的主机晚发射和逐层墙钟时间，再选择没有大空档的采样步。

这套流程用同一会话的原始文件生成四份报告，不需要在集群任务中运行分析器。
