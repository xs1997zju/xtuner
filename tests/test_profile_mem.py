import pickle
import sys
from pathlib import Path

import pytest
import torch


# The profile tools are scripts in profile/, not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profile"))

from mem_ranks import report as ranks_report  # noqa: E402
from mem_snapshot import analyze, load_config, report  # noqa: E402


GIB = 1024**3
COMPUTE, COMM = 0, 7
LAYER = {"filename": "/repo/xtuner/v1/module/decoder_layer/moe_decoder_layer.py", "line": 10, "name": "forward"}
DEEPEP = [
    {"filename": "/env/site-packages/deep_ep/buffer.py", "line": 379, "name": "dispatch"},
    {"filename": "/repo/xtuner/v1/ops/comm/deepep_op.py", "line": 209, "name": "dispatch_forward"},
]
GEMM = [
    {"filename": "/env/site-packages/adaptive_gemm/m_grouped_gemm_varlen.py", "line": 184, "name": "gemm"},
    {"filename": "/repo/xtuner/v1/float8/float8_gmm_tile_wise.py", "line": 367, "name": "forward"},
]


class _Snapshot:
    """Builds a minimal allocator snapshot: events, xtuner ranges and the segment table at the end."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.annotations: list[dict] = []

    def event(self, action: str, t: float, addr: int, gib: float, stream: int = COMPUTE, frames=None) -> None:
        self.events.append(
            {
                "action": action,
                "time_us": t,
                "addr": addr,
                "size": int(gib * GIB),
                "stream": stream,
                "frames": frames or [LAYER],
            }
        )

    def range(self, name: str, start: float, end: float) -> None:
        self.annotations += [
            {"stage": "START", "name": name, "device": 0, "time_us": start},
            {"stage": "END", "name": name, "device": 0, "time_us": end},
        ]

    def dump(self, path: Path, segments: list[dict]) -> Path:
        path.write_bytes(
            pickle.dumps(
                {
                    "device_traces": [self.events],
                    "segments": segments,
                    "external_annotations": self.annotations,
                    "allocator_settings": {},
                }
            )
        )
        return path


def _one_step(tmp_path: Path) -> Path:
    """One forward and one backward of one layer.

    A 1 GiB block lives across steps.  The forward saves 1 GiB (``B``).  In the backward a 2 GiB DeepEP buffer ``C``
    is freed on the host but stays pending on the comm stream; the next 2 GiB request ``E`` then finds nothing, and
    the allocator runs release_cached_blocks: it waits for ``C`` (free_completed), unmaps it and maps ``E`` again.
    """
    s = _Snapshot()
    s.range("xtuner.L1.forward.0", 10, 50)
    s.range("xtuner.L2.layer0.forward", 20, 40)
    s.range("xtuner.L1.backward.0", 60, 100)
    s.range("xtuner.L2.layer0.backward", 61, 99)
    s.event("alloc", 21, addr=0x2000, gib=1)  # B: saved by the forward
    s.event("segment_map", 62, addr=0x9000, gib=2, stream=COMM, frames=DEEPEP)
    s.event("alloc", 62, addr=0x9000, gib=2, stream=COMM, frames=DEEPEP)  # C
    s.event("free_requested", 63, addr=0x9000, gib=2, stream=COMM)  # C pending: record_stream(compute)
    s.event("alloc", 70, addr=0x3000, gib=2, frames=GEMM)  # D
    s.event("free_requested", 75, addr=0x3000, gib=2)
    s.event("free_completed", 75, addr=0x3000, gib=2)
    s.event("free_completed", 79, addr=0x9000, gib=2, stream=COMM)  # the release waits for C
    s.event("segment_unmap", 80, addr=0x9000, gib=2, stream=COMM, frames=GEMM)
    s.event("segment_map", 81, addr=0x5000, gib=2, frames=GEMM)
    s.event("alloc", 81, addr=0x5000, gib=2, frames=GEMM)  # E: the request that ran out of memory
    for t, addr, gib in ((90, 0x5000, 2), (95, 0x2000, 1)):
        s.event("free_requested", t, addr=addr, gib=gib)
        s.event("free_completed", t, addr=addr, gib=gib)
    segments = [
        {
            "device": 0,
            "stream": COMPUTE,
            "address": 0x1000,
            "total_size": 6 * GIB,
            "blocks": [{"size": GIB, "state": "active_allocated"}, {"size": 5 * GIB, "state": "inactive"}],
        }
    ]
    return s.dump(tmp_path / "rank0_memory_snapshot.pickle", segments)


class TestMemSnapshot:
    def test_counters_peaks_and_phases(self, tmp_path):
        # 起止计数器由段表回推；active 峰值在 backward，此时 C 虽已被 host 释放，仍计入 active。
        a = analyze(_one_step(tmp_path), load_config())
        assert a.start == pytest.approx({"allocated": 1, "active": 1, "reserved": 4})  # E's segment is mapped later
        assert a.end == pytest.approx({"allocated": 1, "active": 1, "reserved": 6})
        peaks = a.peaks.set_index("峰值")
        assert peaks.loc["active", "active"] == pytest.approx(6)
        assert peaks.loc["active", "pending"] == pytest.approx(2)
        assert (peaks.loc["active", "L1"], peaks.loc["active", "L2"]) == ("backward.0", "layer0.backward")
        l1 = a.l1.set_index("L1")
        assert l1.loc["forward.0", "allocated 结束"] - l1.loc["forward.0", "allocated 开始"] == pytest.approx(1)
        assert a.l2.set_index("L2").loc["layer0.forward", "结束时净增 allocated（均值）"] == pytest.approx(1)

    def test_active_peak_composition(self, tmp_path):
        # 峰值构成：常驻块、forward 保存的激活、pending 的 DeepEP 缓冲区、按调用点识别的 MoE 块。
        a = analyze(_one_step(tmp_path), load_config())
        by_category = a.composition["category"].set_index("category").GiB
        assert by_category.to_dict() == pytest.approx(
            {"before step": 1, "activation (forward)": 1, "deepep buffer": 2, "moe routed tokens": 2}
        )

    def test_pending_free_and_release(self, tmp_path):
        # release 的触发请求是 E；开头同步放出的正是 pending 的 C，并归到 DeepEP 的调用点和池。
        a = analyze(_one_step(tmp_path), load_config())
        assert a.totals["releases"] == 1
        release = a.releases.iloc[0]
        assert release.trigger == "m_grouped_gemm_varlen.py:184:gemm"
        assert release.request_gib == pytest.approx(2)
        assert release.pending_freed_gib == pytest.approx(2)
        assert release.pending_sources.startswith("buffer.py:379:dispatch")
        assert release.unmapped_gib == pytest.approx(2)
        assert (release.L1, release.L2) == ("backward.0", "layer0.backward")
        pending = a.pending.iloc[0]
        assert (pending.pool, pending.caller, pending["次数"]) == (
            "deepep comm",
            "deepep_op.py:209:dispatch_forward",
            1,
        )
        assert pending["等待中位_ms"] == pytest.approx(16 / 1e3)

    def test_reports_render(self, tmp_path):
        # 两份报告都能生成，且包含关键结论所在的小节。
        path = _one_step(tmp_path)
        cfg = load_config()
        text = report(analyze(path, cfg), cfg, path)
        assert "## 7. release_cached_blocks" in text and "pie showData" in text and "xychart-beta" in text
        ranks = ranks_report(tmp_path, [0], cfg, log_dir=None, workers=1)
        assert "| 0 |" in ranks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestMemSnapshotGPU:
    def test_real_release_is_attributed(self, tmp_path):
        # 真实 snapshot：record_stream 的块在 GPU 追上之前一直 pending，显存上限处的下一次同样大小的分配
        # 触发 release；报告应在 L2 区间里找到它，并指出同步放出的 pending 块。
        size = GIB // 2
        compute, side = torch.cuda.current_stream(), torch.cuda.Stream()
        # Load the kernels first: CUDA lazy loading blocks the host at the first launch behind running work.
        warm = torch.empty(1024, dtype=torch.uint8, device="cuda")
        for stream in (compute, side):
            with torch.cuda.stream(stream):
                warm.fill_(1)
                torch.cuda._sleep(1000)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction((torch.cuda.memory_reserved() + 1.5 * size) / total)
        torch.cuda.memory._record_memory_history(max_entries=10_000, stacks="python")
        try:
            with torch.autograd.profiler.record_function("xtuner.L1.backward.0"):
                with torch.autograd.profiler.record_function("xtuner.L2.layer0.backward"):
                    torch.cuda._sleep(500_000_000)  # the GPU is busy; the host runs ahead
                    x = torch.empty(size, dtype=torch.uint8, device="cuda")
                    side.wait_stream(compute)
                    with torch.cuda.stream(side):
                        x.fill_(1)
                    x.record_stream(side)
                    del x
                    y = torch.empty(size, dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            path = tmp_path / "rank0_memory_snapshot.pickle"
            path.write_bytes(pickle.dumps(torch.cuda.memory._snapshot()))
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
            torch.cuda.set_per_process_memory_fraction(1.0)
        del y
        a = analyze(path, load_config())
        assert a.totals["releases"] == 1
        release = a.releases.iloc[0]
        assert release.L2 == "layer0.backward"
        assert release.pending_freed_gib == pytest.approx(0.5)
        assert release.request_gib == pytest.approx(0.5)
        assert release.pool_largest_free_gib < release.request_gib  # the pool had no block large enough
        assert a.totals["pending_frees"] >= 1
