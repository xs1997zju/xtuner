#!/usr/bin/env python3
"""One rank, one training step: where the GPU time goes, from training phases down to operators.

Reads the tables written by ``trace_extract.py`` and writes a Markdown report (tables and Mermaid charts):

1. Step budget: an exact partition of the step's GPU span into L3 blocks, exposed communication and idle time.
2. L1 training phases and L2 layers / modules: wall clock versus GPU busy time, with Gantt charts.
3. L3 blocks and L4 operators inside the most expensive layer groups, with TFLOP/s and % of peak for the ops whose
   FLOPs ``profile_config.toml`` can count, and a Gantt chart of one representative layer instance.
4. Hot kernels with their launch grid: a slow outlier (e.g. a reduction whose grid equals a hidden size) shows here.
5. Idle gaps on the critical path, charged to the kernel that waited: the collective it waited for, the FSDP module
   being unsharded, whether the host launched it late, and the host call that was running meanwhile.
6. Host-side waits: synchronize calls, ``.item()`` and slow allocator calls by training phase and layer.

"Work" means compute and DeepEP kernels.  In XTuner's synchronous DeepEP path the compute stream waits for DeepEP, so
the union of work kernels is the critical path; NCCL and memcpy time only counts where no work runs ("exposed").

Usage::

    python profile/analyze_step.py <pq_dir> --rank 0 [--out report.md] [--html timeline.html]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from trace_extract import load_config, load_rank, md_table, merge, union_s


WORK = ("compute", "deepep")
MOE_BLOCKS = {"moe_gemm", "moe_comm", "moe_barrier", "moe_route"}
PHASES = ("forward", "recompute", "backward")


def step_partition(gpu: pd.DataFrame) -> pd.Series:
    """Seconds of the step's GPU span by what the GPU was doing: work by L3 block, else exposed comm, else idle.

    Where work kernels overlap (a side stream next to the compute stream), the kernel on the busiest compute stream
    wins; concurrent exposed collectives share the time evenly.  The categories add up to the span exactly.
    """
    main_stream = gpu[gpu.resource == "compute"].groupby("stream").dur.sum().idxmax()
    # The leading digit ranks the candidates for one instant: main-stream work, other work, NCCL, memcpy.
    cats = np.where(
        gpu.resource.isin(WORK),
        np.where(gpu.stream == main_stream, "0", "1") + gpu.l3,
        np.where(gpu.resource == "nccl", "2exposed " + gpu.l3, "3exposed memcpy"),
    )
    events = sorted([(t, 1, c) for t, c in zip(gpu.ts, cats)] + [(t, -1, c) for t, c in zip(gpu.end, cats)])
    active: Counter[str] = Counter()
    seconds: Counter[str] = Counter()
    prev = events[0][0]
    for t, delta, cat in events:
        dt = (t - prev) / 1e6
        if dt > 0:
            live = sorted(c for c, n in active.items() if n > 0)
            top = [c for c in live if c[0] == live[0][0]] if live else []
            share = top if top and top[0][0] == "2" else top[:1]
            for c in share:
                seconds[c[1:]] += dt / len(share)
            if not live:
                seconds["idle"] += dt
        prev = t
        active[cat] += delta
    return pd.Series(seconds).sort_values(ascending=False)


def critical_gaps(gpu: pd.DataFrame) -> pd.DataFrame:
    """Idle intervals of the work union, each charged to the work kernel that ran next."""
    work = gpu[gpu.resource.isin(WORK)].sort_values("ts").reset_index(drop=True)
    busy_until = np.maximum.accumulate(work.end.to_numpy())
    idx = np.where(work.ts.to_numpy()[1:] > busy_until[:-1])[0]
    nxt = work.iloc[idx + 1]
    gaps = pd.DataFrame(
        {
            "start": busy_until[idx],
            "end": nxt.ts.to_numpy(),
            "phase": nxt.phase.to_numpy(),
            "fsdp": nxt.fsdp.to_numpy(),
            "l2": nxt.l2.to_numpy(),
            "launch": nxt.launch_ts.to_numpy(),
        }
    )
    gaps["len"] = gaps.end - gaps.start
    # The GPU waited for the host if the next kernel was launched after the gap began.
    gaps["host_late"] = (gaps.launch - gaps.start) > 0.5 * gaps.len
    resident = pd.Series("", index=gaps.index)
    other = gpu[~gpu.resource.isin(WORK)]
    for tag, sub in other.groupby(np.where(other.resource == "nccl", other.l4, "memcpy")):
        s, e = merge(sub.ts.to_numpy(), sub.end.to_numpy())
        j = np.searchsorted(s, gaps.end.to_numpy()) - 1
        hit = (j >= 0) & (e[np.clip(j, 0, None)] > gaps.start.to_numpy())
        resident[hit] = resident[hit] + "+" + tag
    gaps["resident"] = resident.str.lstrip("+").replace("", "nothing")
    gaps["waits_on"] = [_waits_on(f, l2, r) for f, l2, r in zip(gaps.fsdp, gaps.l2, gaps.resident)]
    return gaps


def longest_gaps(gaps: pd.DataFrame, host: pd.DataFrame, t0: float, n: int) -> pd.DataFrame:
    """The ``n`` longest gaps with the longest host wait (sync / slow allocation) that overlaps each of them."""
    rows = []
    for g in gaps.nlargest(n, "len").itertuples():
        over = host[(host.ts < g.end) & (host.ts + host.dur > g.start)]
        top = over.nlargest(1, "dur")
        rows.append(
            {
                "起点 s": (g.start - t0) / 1e6,
                "时长 ms": g.len / 1e3,
                "阶段": g.phase,
                "在等什么": g.waits_on,
                "主机晚发射": "是" if g.host_late else "否",
                "同时段最长主机等待": ""
                if top.empty
                else f"{top['name'].iloc[0]} {top.dur.iloc[0] / 1e3:.0f}ms @ {_short_l2(top.l2.iloc[0], top.fsdp.iloc[0])}",
            }
        )
    return pd.DataFrame(rows)


def l1_table(gpu: pd.DataFrame) -> pd.DataFrame:
    """One row per L1 range: GPU window (first to last kernel launched in it) and work busy time inside it."""
    rows = []
    for l1, sub in gpu[gpu.l1 != ""].groupby("l1"):
        work = sub[sub.resource.isin(WORK)]
        rows.append(
            {
                "l1": l1.removeprefix("xtuner.L1."),
                "start": sub.ts.min(),
                "end": sub.end.max(),
                "busy": union_s(work.ts.to_numpy(), work.end.to_numpy()),
            }
        )
    return pd.DataFrame(rows).sort_values("start", ignore_index=True)


def l2_instances(gpu: pd.DataFrame) -> pd.DataFrame:
    """One row per L2 range instance: kind (dense / moe / mtp layer, embed_tokens, loss, ...), window and busy time.

    ``busy`` is the time any work kernel runs inside the instance's window, including kernels other threads launch
    meanwhile (the chunked loss runs its gradient GEMMs outside the loss range), so ``wall - busy`` is waiting.
    ``wall`` and ``busy`` exclude nested L2 instances (the MTP wrapper contains its decoder layer), so walls add up.
    """
    work_all = gpu[gpu.resource.isin(WORK)]
    ws, we = merge(work_all.ts.to_numpy(), work_all.end.to_numpy())
    covered = np.r_[0.0, np.cumsum(we - ws)]

    def busy_in(a: float, b: float) -> float:
        i, j = np.searchsorted(we, a, side="right"), np.searchsorted(ws, b, side="left")
        if i >= j:
            return 0.0
        return (covered[j] - covered[i] - max(0.0, a - ws[i]) - max(0.0, we[j - 1] - b)) / 1e6

    rows = []
    for l2_id, sub in gpu[gpu.l2 != ""].groupby("l2_id"):
        name, outer = sub.l2.iloc[0], sub.l2_outer.iloc[0]
        work = sub[sub.resource.isin(WORK)]
        span = work if len(work) else sub
        base = name.removeprefix("xtuner.L2.").rsplit(".", 1)[0]
        match = re.fullmatch(r"layer(\d+)", base)
        if match:
            kind = (
                "mtp layer"
                if outer.startswith("xtuner.L2.mtp")
                else ("moe layer" if sub.l3.isin(MOE_BLOCKS).any() else "dense layer")
            )
        else:
            kind = re.sub(r"\d+$", "", base)
        rows.append(
            {
                "id": l2_id,
                "name": name,
                "outer": outer,
                "base": base,
                "kind": kind,
                "layer": int(match.group(1)) if match else -1,
                "phase": sub.phase.iloc[0],
                "l1": sub.l1.iloc[0],
                "start": span.ts.min(),
                "end": span.end.max(),
                "busy": busy_in(span.ts.min(), span.end.max()),
            }
        )
    inst = pd.DataFrame(rows)
    inst["wall"] = (inst.end - inst.start) / 1e6
    for child in inst[inst.outer != inst.name].itertuples():
        parent = inst[(inst.name == child.outer) & (inst.start <= child.start) & (inst.end >= child.end)]
        if len(parent):
            inst.loc[parent.index[0], ["wall", "busy"]] -= [child.wall, child.busy]
    return inst


def l2_table(inst: pd.DataFrame, span_s: float) -> pd.DataFrame:
    """L2 instances aggregated by (kind, phase)."""
    table = inst.groupby(["kind", "phase"]).agg(n=("wall", "size"), wall=("wall", "sum"), busy=("busy", "sum"))
    table = table.reset_index().sort_values("wall", ascending=False, ignore_index=True)
    return pd.DataFrame(
        {
            "L2": table.kind + " · " + table.phase,
            "实例": table.n,
            "平均墙钟 ms": 1e3 * table.wall / table.n,
            "平均 work 忙 ms": 1e3 * table.busy / table.n,
            "合计墙钟 s": table.wall,
            "占 step %": 100 * table.wall / span_s,
        }
    )


def l2_bars(inst: pd.DataFrame, max_bars: int) -> list[tuple[str, str, float, float]]:
    """Gantt bars for the first micro-batch; consecutive layers of one kind merge when there are too many bars."""
    first = inst[inst.l1.str.endswith(".0")].sort_values("start")
    bars = [
        [r.phase, r.base + (f" {r.kind.split()[0]}" if r.layer >= 0 else ""), r.start, r.end, r.kind, 1]
        for r in first.itertuples()
    ]
    if len(bars) > max_bars:
        merged: list[list] = []
        for bar in bars:
            if merged and merged[-1][0] == bar[0] and merged[-1][4] == bar[4] and bar[4].endswith("layer"):
                merged[-1][3], merged[-1][5] = bar[3], merged[-1][5] + 1
                merged[-1][1] = f"{bar[4]} x{merged[-1][5]}"
            else:
                merged.append(bar)
        bars = merged
    return [(b[0], b[1], b[2], b[3]) for b in bars]


def l3_l4_tables(gpu: pd.DataFrame, ids: np.ndarray, top: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-instance mean time by L3 block and by L4 operator for a group of L2 instances.

    GEMM rows group by (op, N, K, M within 2x): routed-token counts change M of a grouped GEMM from call to call,
    while distinct projections that share N and K keep separate rows.
    """
    sub = gpu[gpu.l2_id.isin(ids)]
    n = len(ids)
    l3 = sub.groupby("l3").dur.sum().sort_values(ascending=False)
    l3_table = pd.DataFrame(
        {"L3": l3.index, "每实例 ms": l3.to_numpy() / n / 1e3, "占比 %": 100 * l3.to_numpy() / l3.sum()}
    )
    parts = sub.mnk.str.split("x", n=1)
    m = pd.to_numeric(parts.str[0], errors="coerce")
    sub = sub.assign(nk=parts.str[1].fillna(""), m=m, m_bucket=np.round(np.log2(m)).fillna(-1))
    l4 = (
        sub.groupby(["l3", "l4", "nk", "m_bucket"])
        .agg(
            calls=("dur", "size"),
            dur=("dur", "sum"),
            flops=("flops", "sum"),
            peak=("peak_tflops", "max"),
            m=("m", "mean"),
        )
        .reset_index()
    )
    l4 = l4.sort_values("dur", ascending=False).head(top)
    tflops = (l4.flops / l4.dur / 1e6).where(l4.flops > 0)
    shape = [f"{m:.0f}x{nk}" if nk else "" for m, nk in zip(l4.m, l4.nk)]
    l4_table = pd.DataFrame(
        {
            "L3": l4.l3,
            "L4": l4.l4,
            "GEMM MxNxK": shape,
            "次/实例": l4.calls / n,
            "每实例 ms": l4.dur / n / 1e3,
            "占比 %": 100 * l4.dur / l3.sum(),
            "TFLOP/s": tflops,
            "占峰值 %": 100 * tflops / l4.peak,
        }
    )
    return l3_table, l4_table


def op_bars(sub: pd.DataFrame, merge_us: float, max_bars: int) -> list[tuple[str, str, float, float]]:
    """Consecutive kernels of the same (L3, L4) merged into Gantt bars; the ``max_bars`` longest are kept."""
    bars: list[list] = []
    for row in sub.sort_values("ts").itertuples():
        if bars and bars[-1][0] == row.l3 and bars[-1][1] == row.l4 and row.ts - bars[-1][3] <= merge_us:
            bars[-1][3] = max(bars[-1][3], row.end)
        else:
            bars.append([row.l3, row.l4, row.ts, row.end])
    keep = sorted(sorted(bars, key=lambda b: b[3] - b[2])[-max_bars:], key=lambda b: b[2])
    return [(b[0], b[1], b[2], b[3]) for b in keep]


def hot_kernels(gpu: pd.DataFrame, top: int) -> pd.DataFrame:
    """Kernels with the most GPU time.  A long median or a grid equal to a tensor dimension marks a slow kernel."""
    work = gpu[gpu.resource.isin(WORK)]
    table = work.groupby("name").agg(
        l4=("l4", "first"),
        l3=("l3", "first"),
        calls=("dur", "size"),
        total=("dur", "sum"),
        p50=("dur", "median"),
        grid=("grid", lambda s: s.mode().iloc[0]),
        block=("block", lambda s: s.mode().iloc[0]),
    )
    table = table.sort_values("total", ascending=False).head(top).reset_index()
    return pd.DataFrame(
        {
            "kernel": table["name"].str.replace(r"^void ", "", regex=True).str.slice(0, 60),
            "L4": table.l4,
            "L3": table.l3,
            "次数": table.calls,
            "合计 s": table.total / 1e6,
            "中位 ms": table.p50 / 1e3,
            "占 work %": 100 * table.total / work.dur.sum(),
            "grid": table.grid,
            "block": table.block,
        }
    )


def host_table(host: pd.DataFrame, stall_ms: float) -> pd.DataFrame:
    """Host waits by kind, training phase and layer; allocator calls only above ``stall_ms``."""
    slow = host[~host.name.str.startswith("aten::empty") | (host.dur >= stall_ms * 1e3)]
    slow = slow.assign(
        phase=[re.sub(r"\.\d+$", "", l1.removeprefix("xtuner.L1.")) or "outside L1" for l1 in slow.l1],
        at=[_short_l2(l2, fsdp) for l2, fsdp in zip(slow.l2, slow.fsdp)],
    )
    table = slow.groupby(["name", "phase", "at"]).dur.agg(["size", "sum", "max"]).reset_index()
    table = table.sort_values("sum", ascending=False, ignore_index=True)
    return pd.DataFrame(
        {
            "主机事件": table["name"],
            "L1": table["phase"],
            "L2 / FSDP": table["at"],
            "次数": table["size"],
            "合计 s": table["sum"] / 1e6,
            "最长 ms": table["max"] / 1e3,
        }
    )


def gantt(title: str, bars: list[tuple[str, str, float, float]]) -> str:
    """Mermaid Gantt of ``(section, label, start_us, end_us)`` bars, in milliseconds from the first bar."""
    t0 = min(b[2] for b in bars)
    lines = [
        "```mermaid",
        "---",
        "config:",
        "  gantt:",
        "    displayMode: compact",
        "---",
        "gantt",
        f"    title {title}",
        "    dateFormat x",
        "    axisFormat %S.%L",
    ]
    for section in dict.fromkeys(b[0] for b in bars):
        lines.append(f"    section {section}")
        for _, label, start, end in (b for b in bars if b[0] == section):
            s, e = int((start - t0) / 1e3), int((end - t0) / 1e3)
            text = re.sub(r"[:#;]", " ", f"{label.replace('::', '.')} {(end - start) / 1e3:.1f}ms")
            lines.append(f"    {text} : {s}, {max(e, s + 1)}")
    return "\n".join(lines + ["```"])


def pie(title: str, seconds: pd.Series) -> str:
    """Mermaid pie; slices under 1% are folded into "other"."""
    total = seconds.sum()
    big = seconds[seconds >= 0.01 * total]
    lines = ["```mermaid", "pie showData", f"    title {title}", *(f'    "{k}" : {v:.2f}' for k, v in big.items())]
    if total > big.sum():
        lines.append(f'    "other" : {total - big.sum():.2f}')
    return "\n".join(lines + ["```"])


def html_timeline(gpu: pd.DataFrame, inst: pd.DataFrame, path: Path) -> None:
    """Interactive Plotly timeline: L1, L2 and one row per L3 block, zoomable down to single kernels."""
    import plotly.graph_objects as go

    t0 = gpu.ts.min()
    rows: dict[str, list[tuple[str, float, float]]] = {
        "L1": [(r.l1, r.start, r.end) for r in l1_table(gpu).itertuples()],
        "L2": [(f"{r.base}.{r.phase} ({r.kind})", r.start, r.end) for r in inst.itertuples()],
    }
    for (l3, l4), sub in gpu.groupby(["l3", "l4"]):
        s, e = merge(sub.ts.to_numpy(), sub.end.to_numpy() + 20)
        rows.setdefault(f"L3 {l3}", []).extend((l4, a, b - 20) for a, b in zip(s, e))
    fig = go.Figure()
    for row, bars in rows.items():
        fig.add_trace(
            go.Bar(
                y=[row] * len(bars),
                x=[(e - s) / 1e3 for _, s, e in bars],
                base=[(s - t0) / 1e3 for _, s, e in bars],
                orientation="h",
                customdata=[label for label, _, _ in bars],
                name=row,
                hovertemplate="%{customdata}<br>%{x:.3f} ms<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="overlay",
        xaxis_title="ms",
        height=150 + 30 * len(rows),
        showlegend=False,
        yaxis={"categoryorder": "array", "categoryarray": list(rows)[::-1]},
    )
    fig.write_html(path, include_plotlyjs="cdn")


def report(pq_dir: Path, rank: int, cfg: dict) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Build the Markdown report; also return the GPU table and L2 instances for ``--html``."""
    opts = cfg["report"]
    gpu, host = load_rank(pq_dir, rank, cfg)
    if not gpu.l2.ne("").any():
        raise ValueError("Trace has no xtuner.L2 ranges; collect a new trace with leveled profiling enabled")
    t0, span = gpu.ts.min(), (gpu.end.max() - gpu.ts.min()) / 1e6
    work = gpu[gpu.resource.isin(WORK)]
    busy = union_s(work.ts.to_numpy(), work.end.to_numpy())
    part = step_partition(gpu)
    gaps = critical_gaps(gpu)
    inst = l2_instances(gpu)
    out = [
        f"# rank{rank} step 时间分析",
        "",
        f"GPU 时间跨度 `{span:.3f}s`；work（compute + DeepEP）忙 `{busy:.3f}s`（`{100 * busy / span:.1f}%`），"
        f"其余 `{span - busy:.3f}s` 是暴露的通信或空闲。",
        "",
    ]

    out += [
        "## 1. 一步时间构成",
        "",
        "对 GPU 时间跨度做精确划分：有 work 在跑的时刻记到该 kernel 的 L3 块；没有 work 时记到驻留的通信"
        "（exposed）或空闲。`moe_barrier` 是 DeepEP notify kernel，时长基本是在等 EP 组里最慢的 rank。",
        "",
        md_table(pd.DataFrame({"类别": part.index, "秒": part.to_numpy(), "占 step %": 100 * part.to_numpy() / span})),
        "",
        pie(f"rank{rank} step {span:.2f}s", part),
        "",
    ]
    phase = (
        work.assign(phase=work.phase.where(work.phase.isin(PHASES), "other")).pivot_table(
            index="l3", columns="phase", values="dur", aggfunc="sum", fill_value=0
        )
        / 1e6
    )
    phase = phase[[c for c in (*PHASES, "other") if c in phase.columns]]
    phase["合计"] = phase.sum(axis=1)
    phase = phase.sort_values("合计", ascending=False)
    phase.loc["合计"] = phase.sum()
    out += [
        "各 L3 块在各阶段的 kernel 时间（秒，work kernel 求和；other = 优化器、clip 等）：",
        "",
        md_table(phase.reset_index().rename(columns={"l3": "L3"})),
        "",
    ]

    l1 = l1_table(gpu)
    wall = (l1.end - l1.start) / 1e6
    out += [
        "## 2. L1 训练阶段",
        "",
        "墙钟 = 该阶段发射的首个到末个 kernel 的跨度；忙占比低说明阶段内有暴露等待。",
        "",
        md_table(
            pd.DataFrame(
                {
                    "L1": l1.l1,
                    "起点 s": (l1.start - t0) / 1e6,
                    "墙钟 s": wall,
                    "work 忙 s": l1.busy,
                    "忙占比 %": 100 * l1.busy / wall,
                }
            ),
            3,
        ),
        "",
        gantt(f"rank{rank} L1", [("L1", r.l1, r.start, r.end) for r in l1.itertuples()]),
        "",
    ]

    layers = inst[inst.layer >= 0].pivot_table(index=["layer", "kind"], columns="phase", values="wall", aggfunc="mean")
    layers = (1e3 * layers[[c for c in PHASES if c in layers.columns]]).reset_index()
    out += [
        "## 3. L2 层与模块",
        "",
        "墙钟 = 实例首个到末个 work kernel 的跨度；work 忙 = 这段时间里任何 work kernel 在跑的时间，包括其他线程"
        "同时发射的（例如分块 loss 在前向里算的梯度 GEMM）。两者都不含嵌套的子 L2（MTP 包装层不含其内部的 decoder 层）。"
        "墙钟明显大于 work 忙的组，内部有暴露等待。",
        "",
        md_table(l2_table(inst, span)),
        "",
        "每层每实例平均墙钟（ms），用来找异常慢的层：",
        "",
        md_table(layers, 1),
        "",
        gantt(f"rank{rank} L2（第一个 micro-batch）", l2_bars(inst, opts["gantt_max_bars"])),
        "",
    ]

    out += [
        "## 4. L3/L4 层内算子",
        "",
        f"取合计墙钟最大的 {opts['l3_groups']} 个层组，表中为每实例平均。GEMM 行按 (算子, N, K) 聚合、M 取平均"
        "（M 相差 2 倍以上的分开列）；"
        "TFLOP/s 只对 profile_config.toml 能算 FLOPs 的算子给出。Gantt 画的是该组墙钟居中的实例。",
        "",
    ]
    groups = inst[inst.kind.str.endswith("layer")].groupby(["kind", "phase"]).wall.sum().sort_values(ascending=False)
    for (kind, ph), total in groups.head(opts["l3_groups"]).items():
        ids = inst[(inst.kind == kind) & (inst.phase == ph)].id.to_numpy()
        l3, l4 = l3_l4_tables(gpu, ids, opts["top_l4"])
        rep = inst[inst.id.isin(ids)].sort_values("wall").iloc[len(ids) // 2]
        sub = gpu[(gpu.l2_id == rep["id"]) & (gpu.resource != "memcpy")]
        out += [
            f"### {kind} · {ph}（{len(ids)} 个实例，合计墙钟 {total:.3f}s）",
            "",
            md_table(l3),
            "",
            md_table(l4),
            "",
            gantt(
                f"{rep['base']}.{ph}（墙钟 {rep['wall'] * 1e3:.1f}ms）",
                op_bars(sub, opts["gantt_merge_us"], opts["gantt_max_bars"]),
            ),
            "",
        ]

    out += ["## 5. 热点 kernel", "", md_table(hot_kernels(gpu, opts["top_kernels"]), 3), ""]

    small = gaps.len < opts["gap_min_ms"] * 1e3
    table = (
        gaps[~small]
        .groupby(["phase", "waits_on"])
        .agg(sec=("len", "sum"), n=("len", "size"), late=("host_late", "sum"))
        .reset_index()
    )
    table = table.sort_values("sec", ascending=False, ignore_index=True)
    table.loc[len(table)] = [
        "*",
        f"< {opts['gap_min_ms']:g} ms 的零碎空闲",
        gaps[small].len.sum(),
        int(small.sum()),
        int(gaps[small].host_late.sum()),
    ]
    out += [
        "## 6. 关键路径空闲",
        "",
        f"work 并集之外共 `{gaps.len.sum() / 1e6:.3f}s`。每段空闲记到其后第一个 work kernel："
        "`[...]` 是空闲时驻留的通信，`X AG` 表示在等 FSDP 解分片 X；主机晚发射 = 下一个 kernel 在空闲开始后才 launch，"
        "问题在 CPU 侧。",
        "",
        md_table(
            pd.DataFrame(
                {
                    "阶段": table.phase,
                    "在等什么": table.waits_on,
                    "秒": table.sec / 1e6,
                    "次数": table.n,
                    "其中主机晚发射次数": table.late,
                }
            ),
            3,
        ),
        "",
        "最长的几段空闲，以及同一时段最长的主机等待：",
        "",
        md_table(longest_gaps(gaps, host, t0, 8), 3),
        "",
    ]

    out += [
        "## 7. 主机侧等待",
        "",
        f"同步调用与慢分配（`aten::empty*` ≥ {opts['host_stall_ms']:g}ms，通常是显存触顶时分配器在释放/重映射缓存）。"
        "`aten::_local_scalar_dense` 即 `.item()`，内部也会 cudaStreamSynchronize，两行有重叠。",
        "",
        md_table(host_table(host, opts["host_stall_ms"]).head(15), 3),
        "",
    ]
    return "\n".join(out), gpu, inst


def _waits_on(fsdp: str, l2: str, resident: str) -> str:
    for module in ("embed_tokens", "lm_head", "mtp_block", "norm"):
        if module in fsdp:
            return f"{module} AG"
    if "all_gather_copy_out (layers" in fsdp:
        return "layer AG"
    return f"{re.sub(r'[0-9]+', 'N', l2.removeprefix('xtuner.L2.')) or 'no L2'} [{resident}]"


def _short_l2(l2: str, fsdp: str) -> str:
    where = re.sub(r"[0-9]+", "N", l2.removeprefix("xtuner.L2.")) or "-"
    return f"{where} / {fsdp.removeprefix('FSDP::')}" if fsdp else where


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pq_dir", type=Path, help="output directory of trace_extract.py")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--config", type=Path, help="rules file, defaults to profile/profile_config.toml")
    parser.add_argument("--out", type=Path, help="write the Markdown report here instead of stdout")
    parser.add_argument("--html", type=Path, help="also write an interactive Plotly timeline")
    args = parser.parse_args()
    text, gpu, inst = report(args.pq_dir, args.rank, load_config(args.config))
    if args.out:
        args.out.write_text(text)
    else:
        print(text)
    if args.html:
        html_timeline(gpu, inst, args.html)


if __name__ == "__main__":
    main()
