import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# The profile tools are scripts in profile/, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profile"))

from analyze_ranks import collective_table, deepep_tables, lateness_table  # noqa: E402
from analyze_step import critical_gaps, l2_instances, step_partition  # noqa: E402
from analyze_step import report as step_report
from trace_extract import extract, load_config, load_rank  # noqa: E402


MAIN, AUTOGRAD = 1, 2


class _Trace:
    """Builds a minimal torch-profiler Chrome trace: ranges, CPU ops and the kernels they launch."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.next_id = 100

    def range(self, name: str, tid: int, ts: float, dur: float) -> None:
        self.events.append({"ph": "X", "cat": "user_annotation", "name": name, "tid": tid, "ts": ts, "dur": dur})

    def host(self, name: str, tid: int, ts: float, dur: float, cat: str = "cpu_op") -> None:
        self.events.append({"ph": "X", "cat": cat, "name": name, "tid": tid, "ts": ts, "dur": dur, "args": {}})

    def kernel(
        self,
        name: str,
        *,
        op: str,
        tid: int,
        launch: float,
        ts: float,
        dur: float,
        stream: int = 7,
        dims: list | None = None,
        **args: object,
    ) -> None:
        self.next_id += 1
        ext = corr = self.next_id
        self.events.append(
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": op,
                "tid": tid,
                "ts": launch - 1,
                "dur": 2,
                "args": {"External id": ext, "Input Dims": dims or []},
            }
        )
        self.events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "tid": tid,
                "ts": launch,
                "dur": 1,
                "args": {"correlation": corr},
            }
        )
        self.events.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": name,
                "tid": stream,
                "ts": ts,
                "dur": dur,
                "args": {"External id": ext, "correlation": corr, "stream": stream, **args},
            }
        )

    def load(self, tmp_path: Path, rank: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
        trace = tmp_path / f"rank{rank}_trace_time-1s.json"
        trace.write_text(json.dumps({"traceEvents": self.events}))
        extract(trace, tmp_path / "pq")
        return load_rank(tmp_path / "pq", rank, load_config())


def _layer_step() -> _Trace:
    """One forward and one backward (with recompute) of layer1, plus an FSDP all-gather."""
    t = _Trace()
    t.range("xtuner.L1.forward.0", MAIN, 0, 100)
    t.range("xtuner.L1.backward.0", MAIN, 100, 200)
    t.range("xtuner.L2.layer1.forward", MAIN, 10, 50)
    t.range("xtuner.L2.layer1.backward", AUTOGRAD, 150, 130)
    t.range("xtuner.L2.layer1.recompute", AUTOGRAD, 160, 40)
    t.kernel(
        "adaptive_gemm::fp8_gemm_kernel<256u, 64u>",
        op="moe::gemm_fp8_fp8_bf16_nt",
        tid=MAIN,
        launch=21,
        ts=30,
        dur=10,
        dims=[[[128, 64], [128, 1]], [[256, 64], [2, 1]], [128, 256]],
    )
    t.kernel(
        "ncclDevKernel_AllGather_RING_LL",
        op="record_param_comms",
        tid=MAIN,
        launch=51,
        ts=45,
        dur=45,
        stream=20,
        **{
            "Collective name": "_allgather_base",
            "Process Group Description": "mesh_default.fsdp",
            "Process Group Ranks": "[0, 1]",
            "Group size": 2,
            "In msg nelems": 8,
            "Out msg nelems": 16,
            "dtype": "BFloat16",
        },
    )
    t.kernel(
        "triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_view_5",
        op="triton_red_fused__fused_rms_norm_backward__to_copy_add_mul_view_5",
        tid=AUTOGRAD,
        launch=171,
        ts=175,
        dur=5,
    )
    t.kernel(
        "void deep_ep::intranode::cached_notify_combine<8>(void**)",
        op="DeepEPCombine",
        tid=AUTOGRAD,
        launch=221,
        ts=230,
        dur=10,
        stream=26,
    )
    return t


class TestTraceExtract:
    def test_glm53_kda_kernel_is_attention(self, tmp_path: Path):
        trace = _Trace()
        trace.range("xtuner.L2.layer1.forward", MAIN, 0, 100)
        trace.kernel("chunk_kda_fwd", op="xtuner_kda::chunk_kda_fwd", tid=MAIN, launch=10, ts=20, dur=5)
        gpu, _ = trace.load(tmp_path)
        assert (gpu.iloc[0].l2, gpu.iloc[0].l3) == ("xtuner.L2.layer1.forward", "attention")

    def test_ranges_ops_and_labels_follow_the_launch(self, tmp_path: Path):
        gpu, _ = _layer_step().load(tmp_path)
        gemm, nccl, norm, notify = (gpu.iloc[i] for i in range(4))

        assert (gemm.l1, gemm.l2, gemm.phase) == ("xtuner.L1.forward.0", "xtuner.L2.layer1.forward", "forward")
        assert (gemm.l4, gemm.l3, gemm.mnk) == ("fp8_gemm", "gemm", "128x256x64")
        assert gemm.flops == 2 * 128 * 256 * 64
        # Launched by the autograd thread: L1 comes from the main thread's range at that time, L2 from its own.
        assert (norm.l1, norm.l2, norm.phase) == ("xtuner.L1.backward.0", "xtuner.L2.layer1.recompute", "recompute")
        assert (norm.l4, norm.l3) == ("rmsnorm_bwd (inductor)", "norm")
        assert (nccl.resource, nccl.l4, nccl.l3) == ("nccl", "fsdp AG", "fsdp comm")
        assert (nccl.pg_ranks, nccl.pg_size) == ("[0, 1]", 2)
        assert (notify.resource, notify.l3, notify.phase) == ("deepep", "moe_barrier", "backward")

    def test_host_table_keeps_syncs_and_slow_allocations_only(self, tmp_path: Path):
        t = _layer_step()
        t.host("cudaStreamSynchronize", MAIN, 80, 500, cat="cuda_runtime")
        t.host("aten::empty_strided", AUTOGRAD, 190, 2000)
        t.host("aten::empty", MAIN, 20, 10)
        _, host = t.load(tmp_path)

        assert sorted(host.name) == ["aten::empty_strided", "cudaStreamSynchronize"]
        stall = host[host.name == "aten::empty_strided"].iloc[0]
        assert (stall.l1, stall.l2) == ("xtuner.L1.backward.0", "xtuner.L2.layer1.recompute")


class TestAnalyzeStep:
    def test_report_explains_missing_layer_ranges(self, tmp_path: Path):
        trace = _Trace()
        trace.kernel("plain_kernel", op="aten::add", tid=MAIN, launch=10, ts=20, dur=5)
        trace.load(tmp_path)
        with pytest.raises(ValueError, match="no xtuner.L2 ranges"):
            step_report(tmp_path / "pq", 0, load_config())

    def test_step_partition_splits_the_span_exactly(self, tmp_path: Path):
        gpu, _ = _layer_step().load(tmp_path)
        part = step_partition(gpu)

        # Span 30..240 us: work 10 + 5 + 10, the all-gather is exposed for 45, the rest is idle.
        assert part.sum() == pytest.approx(210e-6)
        assert part["gemm"] == pytest.approx(10e-6)
        assert part["moe_barrier"] == pytest.approx(10e-6)
        assert part["exposed fsdp comm"] == pytest.approx(45e-6)
        assert part["idle"] == pytest.approx(140e-6)

    def test_gaps_are_charged_to_the_kernel_that_waited(self, tmp_path: Path):
        gpu, _ = _layer_step().load(tmp_path)
        gaps = critical_gaps(gpu).sort_values("start")

        first, second = gaps.iloc[0], gaps.iloc[1]
        assert (first.start, first.end, first.phase) == (40, 175, "recompute")
        assert first.waits_on == "layerN.recompute [fsdp AG]"
        # The norm kernel was launched at 171, long after the GPU went idle at 40.
        assert first.host_late
        assert (second.start, second.end, second.resident) == (180, 230, "nothing")

    def test_nested_l2_walls_do_not_double_count(self, tmp_path: Path):
        t = _Trace()
        t.range("xtuner.L1.forward.0", MAIN, 0, 1000)
        t.range("xtuner.L2.mtp1.forward", MAIN, 0, 300)
        t.range("xtuner.L2.layer6.forward", MAIN, 20, 150)
        for launch, ts in ((5, 10), (25, 50), (210, 300)):
            t.kernel("vectorized_elementwise_kernel", op="aten::add_", tid=MAIN, launch=launch, ts=ts, dur=10)
        gpu, _ = t.load(tmp_path)
        inst = l2_instances(gpu).set_index("base")

        assert inst.loc["layer6", "kind"] == "mtp layer"
        assert inst.loc["layer6", "wall"] == pytest.approx(10e-6)
        # The wrapper spans 10..310 us; without the nested layer that leaves 290 us.
        assert inst.loc["mtp1", "wall"] == pytest.approx(290e-6)


class TestAnalyzeRanks:
    @staticmethod
    def _nccl(
        rank: int,
        arrivals: list[float],
        ends: list[float],
        out: list[int],
        members: str = "[0, 1, 2]",
        size: int = 3,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "rank": rank,
                "ts": arrivals,
                "end": ends,
                "dur": np.subtract(ends, arrivals),
                "l4": "fsdp AG",
                "coll": "_allgather_base",
                "pg_ranks": members,
                "pg_size": size,
                "in_nelems": 1,
                "out_nelems": out,
                "dtype": "BFloat16",
            }
        )

    @classmethod
    def _truncated_group(cls, ranks: range) -> pd.DataFrame:
        """Two all-gathers of a 31-rank group, whose rank list the profiler truncates; rank 30 always arrives last."""
        members = "[" + ", ".join(map(str, range(29))) + ", ..., 30]"
        return pd.concat(
            [
                cls._nccl(r, [10.0 + 10 * (r == 30), 110.0 + 10 * (r == 30)], [100.0, 200.0], [2, 4], members, 31)
                for r in ranks
            ]
        )

    def test_lateness_names_the_straggler_and_skips_a_missing_record(self):
        # Three all-gathers told apart by payload; rank 2 lost the record of the second one.
        nccl = pd.concat(
            [
                self._nccl(0, [10.0, 110.0, 210.0], [100.0, 200.0, 300.0], [2, 4, 8]),
                self._nccl(1, [11.0, 111.0, 211.0], [100.0, 200.0, 300.0], [2, 4, 8]),
                self._nccl(2, [20.0, 220.0], [100.0, 300.0], [2, 8]),
            ]
        )
        table, skipped = lateness_table(nccl)

        row = table.iloc[0]
        assert row["集合通信/组"] == 2
        assert row["迟到 p50 ms"] == pytest.approx(9e-3)
        assert row["最常最后到达的 rank（占比）"].startswith("2 (100%)")
        assert skipped == []

    def test_lateness_covers_a_group_whose_rank_list_the_profiler_truncated(self):
        table, skipped = lateness_table(self._truncated_group(range(31)))

        row = table.iloc[0]
        assert row["集合通信/组"] == 2
        assert row["迟到 p50 ms"] == pytest.approx(10e-3)
        assert row["最常最后到达的 rank（占比）"].startswith("30 (100%)")
        assert skipped == []

    def test_lateness_skips_a_group_with_an_unloaded_member(self):
        table, skipped = lateness_table(self._truncated_group(range(30)))

        assert table.empty
        assert skipped == ["fsdp [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11..."]

    def test_collective_waiting_share_uses_the_fastest_instance(self):
        nccl = self._nccl(0, [90.0, 180.0, 270.0], [100.0, 200.0, 300.0], [2, 2, 2])
        row = collective_table(nccl, n_ranks=1).iloc[0]

        # Durations 10, 20, 30 us: the floor is 10, so 30 of the 60 resident microseconds are waiting.
        assert row["floor ms"] == pytest.approx(0.01)
        assert row["等待占比 %"] == pytest.approx(50.0)

    def test_deepep_wait_is_the_excess_over_the_fastest_rank(self):
        names = [
            "deep_ep::intranode::dispatch<2>",
            "deep_ep::intranode::cached_notify_combine<2>",
            "deep_ep::intranode::combine<2>",
        ]
        # Rank 1 spends 30 us more in the expert phase, so rank 0 waits 30 us in the combine notify.
        frames = [
            pd.DataFrame({"name": names, "ts": [0.0, 50.0, 90.0], "dur": [10.0, 40.0, 5.0], "rank": 0}),
            pd.DataFrame({"name": names, "ts": [0.0, 80.0, 90.0], "dur": [10.0, 10.0, 5.0], "rank": 1}),
        ]
        kernels, phase = deepep_tables(pd.concat(frames, ignore_index=True), {0, 1})

        notify = kernels.set_index("kernel").loc["cached_notify_combine"]
        assert notify["自身耗时 s"] == pytest.approx(10e-6)
        assert notify["等待 s/rank"] == pytest.approx(15e-6)
        assert phase.iloc[0]["corr(专家耗时, 等待)"] == pytest.approx(-1.0)
