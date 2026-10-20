#!/usr/bin/env python3
"""All profiled ranks of one step: who sets the pace, which collectives wait, and how much of DeepEP is waiting.

1. Per-rank budget: work busy time (split into DeepEP barrier waits and the rest), exposed communication, idle time,
   and host stalls.  In a synchronous job the slowest rank sets the pace and every other rank shows the difference
   as waiting, so compare the non-waiting parts across ranks.
2. Collectives grouped by (process group, collective, dtype, payload): the fastest instance is the wire time
   ("floor"); resident time far above count x floor is waiting for peers, not a slow link.
3. Barrier lateness per process group: each group's collectives are matched across its ranks by signature and
   clock-aligned on their end times (a collective ends together everywhere); lateness = last arrival minus median
   arrival, and the rank that is most often last is the straggler.
4. DeepEP per EP group: every kernel split into its own cost (the fastest rank's duration) and waiting, and how much
   of the combine barrier wait is explained by expert-phase imbalance between the EP ranks.
5. The same kernel on every rank: per-rank median duration of the heaviest compute kernels.  One rank slower on all
   of them is a slow GPU; spread only on data-dependent kernels (attention, grouped GEMM) is load imbalance.

Usage::

    python profile/analyze_ranks.py <pq_dir> [--ranks 0 1 2 3] [--out ranks.md]
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from analyze_step import WORK, critical_gaps, step_partition
from trace_extract import load_config, load_rank, md_table, union_s


DTYPE_BYTES = {"BFloat16": 2, "Half": 2, "Float": 4, "Float8_e4m3fn": 1, "Byte": 1, "Char": 1, "Int": 4, "Long": 8}


def rank_summary(pq_dir: Path, rank: int, cfg: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Budget row of one rank, its NCCL and DeepEP kernels, and per-kernel-name compute stats."""
    gpu, host = load_rank(pq_dir, rank, cfg)
    part = step_partition(gpu)
    gaps = critical_gaps(gpu)
    work = gpu[gpu.resource.isin(WORK)]
    busy = union_s(work.ts.to_numpy(), work.end.to_numpy())
    stall = host[host.name.str.startswith("aten::empty") & (host.dur >= cfg["report"]["host_stall_ms"] * 1e3)]
    big = gaps.len >= cfg["report"]["gap_min_ms"] * 1e3
    row = {
        "rank": rank,
        "span s": (gpu.end.max() - gpu.ts.min()) / 1e6,
        "work 忙 s": busy,
        "其中 DeepEP 屏障 s": part.get("moe_barrier", 0.0),
        "非等待计算 s": busy - part.get("moe_barrier", 0.0),
        "暴露通信 s": part[part.index.str.startswith("exposed")].sum(),
        "空闲 s": part.get("idle", 0.0),
        "其中主机晚发射 s": gaps[big & gaps.host_late].len.sum() / 1e6,
        "分配器卡顿 s": stall.dur.sum() / 1e6,
    }
    cols = ["name", "ts", "end", "dur", "l4", "coll", "pg_ranks", "pg_size", "in_nelems", "out_nelems", "dtype"]
    nccl = gpu.loc[gpu.resource == "nccl", cols].assign(rank=rank)
    deepep = gpu.loc[gpu.resource == "deepep", ["name", "ts", "dur"]].sort_values("ts").assign(rank=rank)
    kernels = (
        gpu[gpu.resource == "compute"]
        .groupby("name")
        .agg(l4=("l4", "first"), n=("dur", "size"), total=("dur", "sum"), p50=("dur", "median"))
        .assign(rank=rank)
    )
    return row, nccl, deepep, kernels.reset_index()


def collective_table(nccl: pd.DataFrame, n_ranks: int) -> pd.DataFrame:
    """NCCL kernels by (group verb, dtype, payload): wire-time floor, typical durations and the waiting share."""
    nccl = nccl.assign(
        bytes=np.maximum(nccl.in_nelems.fillna(0), nccl.out_nelems.fillna(0)) * nccl.dtype.map(DTYPE_BYTES).fillna(2)
    )
    table = (
        nccl.groupby(["l4", "dtype", "bytes"])
        .dur.agg(n="size", total="sum", floor="min", p50="median", p90=lambda d: d.quantile(0.9))
        .reset_index()
    )
    table = table[table.total >= 0.005 * table.total.sum()].sort_values("total", ascending=False)
    gb = table["bytes"] / 1e9
    wire = table.n * table.floor
    return pd.DataFrame(
        {
            "集合通信": table.l4,
            "dtype": table.dtype,
            "GB": gb,
            "次/rank": table.n / n_ranks,
            "驻留 s/rank": table.total / n_ranks / 1e6,
            "floor ms": table.floor / 1e3,
            "p50 ms": table.p50 / 1e3,
            "p90 ms": table.p90 / 1e3,
            "floor GB/s": gb / (table.floor / 1e6),
            "p50 GB/s": gb / (table.p50 / 1e6),
            "等待占比 %": 100 * (1 - wire / table.total),
        }
    )


def lateness_table(nccl: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Barrier lateness per process-group kind, over every group whose member ranks are all loaded."""
    rows, skipped = [], []
    nccl = nccl[nccl.pg_ranks.str.startswith("[")]
    for (tag, members), sub in nccl.groupby([nccl.l4.str.split().str[0], "pg_ranks"]):
        # The profiler prints rank lists longer than 30 as "[0, 1, ..., 28, ..., 255]", so membership is taken from
        # the records instead: only members record a group's collectives, so the group is complete once the ranks
        # recording it add up to its size.
        group = sorted(sub["rank"].unique())
        if len(group) < 2 or len(group) != sub.pg_size.iloc[0]:
            skipped.append(f"{tag} {members if len(members) < 40 else members[:37] + '...'}")
            continue
        seqs = {r: s.sort_values("ts") for r, s in sub.groupby("rank")}
        late, last = _lateness(seqs, group)
        if len(late):
            rows.append({"tag": tag, "late": late, "last": last})
    out = []
    for tag, items in pd.DataFrame(rows).groupby("tag") if rows else []:
        late = np.concatenate(items.late.to_list())
        last = np.concatenate(items["last"].to_list())
        top, count = np.unique(last, return_counts=True)
        worst = np.argsort(-count)[:3]
        out.append(
            {
                "进程组": tag,
                "组数": len(items),
                "集合通信/组": len(late) / len(items),
                "迟到合计 s/组": late.sum() / 1e6 / len(items),
                "迟到 p50 ms": np.median(late) / 1e3,
                "迟到 p90 ms": np.percentile(late, 90) / 1e3,
                "最常最后到达的 rank（占比）": ", ".join(
                    f"{top[i]} ({100 * count[i] / len(last):.0f}%)" for i in worst
                ),
            }
        )
    return pd.DataFrame(out), skipped


def deepep_tables(deepep: pd.DataFrame, ranks: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DeepEP kernels split into own cost and waiting, per EP group of contiguous ranks."""
    # The notify kernels are templated on the EP group size, e.g. notify_dispatch<8>; EP groups are contiguous ranks.
    sizes = deepep.name.str.extract(r"notify\w*<(\d+)>", expand=False).dropna()
    size = int(sizes.iloc[0]) if len(sizes) else 0
    kernel_rows, phase_rows = [], []
    for first in range(0, max(ranks) + 1, size or max(ranks) + 1):
        group = list(range(first, first + size))
        if not size or not set(group) <= ranks:
            continue
        seqs = [deepep[deepep["rank"] == r] for r in group]
        n = min(len(s) for s in seqs)
        kinds = seqs[0].name.str.extract(r"deep_ep::\w+::(\w+)", expand=False).to_numpy()[:n]
        if any((s.name.to_numpy()[:n] != seqs[0].name.to_numpy()[:n]).any() for s in seqs):
            continue
        start = np.stack([s.ts.to_numpy()[:n] for s in seqs])
        dur = np.stack([s.dur.to_numpy()[:n] for s in seqs])
        wait = dur - dur.min(axis=0)
        for kind in dict.fromkeys(kinds):
            m = kinds == kind
            kernel_rows.append(
                {
                    "EP 组": f"{group[0]}-{group[-1]}",
                    "kernel": kind,
                    "次数": int(m.sum()),
                    "每 rank 合计 s": dur[:, m].sum(1).mean() / 1e6,
                    "自身耗时 s": dur[:, m].min(0).sum() / 1e6,
                    "等待 s/rank": wait[:, m].sum(1).mean() / 1e6,
                }
            )
        # Expert phase = dispatch end -> start of the next combine notify; its spread is what the notify waits out.
        idx = np.array([i for i in range(n - 2) if kinds[i] == "dispatch" and kinds[i + 1] == "cached_notify_combine"])
        if len(idx):
            expert = (start[:, idx + 1] - (start[:, idx] + dur[:, idx])) / 1e3
            notify = dur[:, idx + 1] / 1e3
            ratio = expert.max(0) / np.maximum(expert.mean(0), 1e-9)
            corr = np.nanmean([np.corrcoef(expert[:, j], notify[:, j])[0, 1] for j in range(len(idx))])
            phase_rows.append(
                {
                    "EP 组": f"{group[0]}-{group[-1]}",
                    "专家阶段次数": len(idx),
                    "平均 ms": expert.mean(),
                    "最慢/平均 p50": np.median(ratio),
                    "最慢/平均 p90": np.percentile(ratio, 90),
                    "Σ(最慢-平均) s": (expert.max(0) - expert.mean(0)).sum() / 1e3,
                    "combine 屏障等待 s/rank": notify.sum(1).mean() / 1e3,
                    "corr(专家耗时, 等待)": corr,
                }
            )
    return pd.DataFrame(kernel_rows), pd.DataFrame(phase_rows)


def kernel_spread(kernels: pd.DataFrame, ranks: list[int], top: int) -> pd.DataFrame:
    """Per-rank median duration of the heaviest compute kernels that run the same number of times on every rank."""
    wide = kernels.pivot_table(index="name", columns="rank", values=["n", "total", "p50"])
    same = wide["n"].nunique(axis=1).eq(1) & wide["n"].notna().all(axis=1)
    heavy = wide.loc[same, "total"].mean(axis=1).sort_values(ascending=False).head(top).index
    p50 = wide.loc[heavy, "p50"] / 1e3
    table = pd.DataFrame(
        {
            "kernel": [n.removeprefix("void ")[:50] for n in heavy],
            "L4": kernels.drop_duplicates("name").set_index("name").loc[heavy, "l4"].to_numpy(),
        }
    )
    for r in ranks:
        table[f"r{r} ms"] = p50[r].to_numpy()
    table["离散度 %"] = (100 * (p50.max(axis=1) - p50.min(axis=1)) / p50.mean(axis=1)).to_numpy()
    table["最慢 rank"] = p50.idxmax(axis=1).to_numpy()
    return table


def xychart(title: str, ranks: list[int], bars: list[float], line: list[float], y: str) -> str:
    """Mermaid bar-plus-line chart over ranks."""
    return "\n".join(
        [
            "```mermaid",
            "xychart-beta",
            f'    title "{title}"',
            f"    x-axis [{', '.join(f'r{r}' for r in ranks)}]",
            f'    y-axis "{y}"',
            f"    bar [{', '.join(f'{v:.2f}' for v in bars)}]",
            f"    line [{', '.join(f'{v:.2f}' for v in line)}]",
            "```",
        ]
    )


def report(pq_dir: Path, ranks: list[int], cfg: dict, workers: int) -> str:
    """Build the Markdown report for ``ranks``."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(rank_summary, [pq_dir] * len(ranks), ranks, [cfg] * len(ranks)))
    budget = pd.DataFrame([r[0] for r in results])
    nccl = pd.concat([r[1] for r in results], ignore_index=True)
    deepep = pd.concat([r[2] for r in results], ignore_index=True)
    kernels = pd.concat([r[3] for r in results], ignore_index=True)
    loaded = set(ranks)
    compute = budget["非等待计算 s"]
    out = [
        f"# {len(ranks)} 个 rank 的对比",
        "",
        f"非等待计算（work 忙减去 DeepEP 屏障等待）最多的是 rank{int(budget.loc[compute.idxmax(), 'rank'])}"
        f"（`{compute.max():.3f}s`），最少 rank{int(budget.loc[compute.idxmin(), 'rank'])}（`{compute.min():.3f}s`），"
        f"差 `{100 * (compute.max() / compute.min() - 1):.1f}%`。",
        "",
    ]

    out += [
        "## 1. 每个 rank 的时间构成",
        "",
        "同步训练里最慢的 rank 决定节奏，其余 rank 把差值表现为等待（DeepEP 屏障、暴露通信、空闲）。"
        "非等待计算 = work 忙 - DeepEP 屏障；暴露通信 + 空闲 = work 之外的时间，其中主机晚发射 = 下一个 work kernel "
        f"在这段时间开始后才 launch 的部分（≥ {cfg['report']['gap_min_ms']:g}ms 的空档，CPU 侧原因）。"
        "比较非等待计算、主机晚发射和分配器卡顿，找出拖慢全局的 rank。",
        "",
        md_table(budget, 3),
        "",
        xychart(
            "每个 rank：非等待计算（柱）与等待合计（线），秒",
            ranks,
            compute.to_list(),
            (budget["其中 DeepEP 屏障 s"] + budget["暴露通信 s"] + budget["空闲 s"]).to_list(),
            "s",
        ),
        "",
    ]

    out += [
        "## 2. 集合通信：线速与等待",
        "",
        "按（进程组 集合通信, dtype, 字节数）分组。floor = 最快一次，近似线速；等待占比 = 1 - 次数 x floor / 驻留。"
        "floor 带宽正常而等待占比高，说明是在等其他 rank，不是链路慢。",
        "",
        md_table(collective_table(nccl, len(ranks)), 2),
        "",
    ]

    late, skipped = lateness_table(nccl)
    out += [
        "## 3. 屏障迟到",
        "",
        "同一进程组的集合通信在各 rank 上按签名配对，用结束时刻对齐时钟；迟到 = 最后到达 - 中位到达。"
        "最常最后到达的 rank 就是拖慢该组的 rank。",
        "",
        md_table(late, 2) if len(late) else "（没有成员齐全的进程组）",
        "",
    ]
    if skipped:
        out += [f"成员未全部加载、未分析的组：{len(skipped)} 个（如 {skipped[0]}）。", ""]

    deep, phase = deepep_tables(deepep, loaded)
    if len(deep):
        out += [
            "## 4. DeepEP：自身耗时与等待",
            "",
            "同一 EP 组内按调用顺序对齐各 rank 的 DeepEP kernel；最快 rank 的耗时近似 kernel 自身代价，其余是等最慢 rank。",
            "",
            md_table(deep, 3),
            "",
            "专家阶段（dispatch 结束到下一次 combine 屏障开始）的负载不均；corr 接近 -1 说明 combine 屏障等待"
            "几乎全部由专家阶段耗时不均造成（路由不均，或该阶段里某个 rank 的主机卡顿）：",
            "",
            md_table(phase, 3),
            "",
        ]

    out += [
        "## 5. 同名 kernel 的跨 rank 耗时",
        "",
        "每个 rank 上次数相同的最重 compute kernel 的中位耗时。某个 rank 在所有 kernel 上都慢，是 GPU 本身慢；"
        "只有注意力、grouped GEMM 这类随数据变化的 kernel 离散，是负载不均。",
        "",
        md_table(kernel_spread(kernels, ranks, cfg["report"]["top_kernels"] // 2), 2),
        "",
    ]
    return "\n".join(out)


def _lateness(seqs: dict[int, pd.DataFrame], group: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Lateness (us) and last-arriving rank of every collective found on all ``group`` ranks.

    Collectives pair by ``(collective, out_nelems)`` with a two-pointer walk, so a record missing on one rank skips
    one barrier instead of shifting every later pairing.
    """
    ref = group[0]
    sigs = {r: list(zip(seqs[r].coll.astype(str), seqs[r].out_nelems.fillna(-1))) for r in group}
    pos = {r: 0 for r in group}
    matched: dict[int, list[int]] = {r: [] for r in group}
    for i, sig in enumerate(sigs[ref]):
        hit = {ref: i}
        for r in group[1:]:
            if pos[r] < len(sigs[r]) and sigs[r][pos[r]] == sig:
                hit[r] = pos[r]
        if len(hit) == len(group):
            for r, j in hit.items():
                matched[r].append(j)
        # Ranks that matched move on; a rank missing this record waits for the next reference collective.
        for r, j in hit.items():
            pos[r] = j + 1
    if not matched[ref]:
        return np.array([]), np.array([])
    ends = {r: seqs[r].end.to_numpy()[matched[r]] for r in group}
    offset = {r: np.median(ends[r] - ends[ref]) for r in group}
    arrival = np.stack([seqs[r].ts.to_numpy()[matched[r]] - offset[r] for r in group])
    return arrival.max(0) - np.median(arrival, axis=0), np.asarray(group)[arrival.argmax(0)]


def _parse_ranks(values: list[str] | None, pq_dir: Path) -> list[int]:
    if not values:
        return sorted(int(re.match(r"rank(\d+)\.gpu", p.name).group(1)) for p in pq_dir.glob("rank*.gpu.parquet"))
    out: list[int] = []
    for v in values:
        lo, _, hi = v.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pq_dir", type=Path, help="output directory of trace_extract.py")
    parser.add_argument("--ranks", nargs="+", help="ranks or ranges such as 0-7 (default: every extracted rank)")
    parser.add_argument("--config", type=Path, help="rules file, defaults to profile/profile_config.toml")
    parser.add_argument("--out", type=Path, help="write the Markdown report here instead of stdout")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    text = report(args.pq_dir, _parse_ranks(args.ranks, args.pq_dir), load_config(args.config), args.workers)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
