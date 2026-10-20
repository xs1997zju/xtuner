#!/usr/bin/env python3
"""All ranks of one memory profile: which rank runs out of memory first, and whether the profiled step is typical.

1. Per rank, from its snapshot: the three peaks, headroom to the device, allocator releases, the largest amount of
   memory pending at once, and a leak check.  A rank with releases stalls its host; the others then wait for it at the
   next collective (see ``analyze_ranks.py`` for that side).
2. Per step, from the training logs of the same run: ``max_memory`` and ``reserved_memory`` across ranks, to check that
   the profiled step is representative and to see the peak of steps that were not profiled.

Usage::

    python profile/mem_ranks.py <profiling_memory/step-N> [--ranks 0 1 ...] [--logs <run>/logs] [--out report.md]
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from mem_snapshot import analyze, load_config
from trace_extract import md_table


LOG_STEP = re.compile(
    r"Step (\d+)/\d+ .*? time: ([0-9.]+) .*?max_memory: ([0-9.]+) GB reserved_memory: ([0-9.]+) GB", re.S
)


def rank_row(path: Path, rank: int, cfg: dict) -> dict:
    """The per-rank summary of one snapshot (no peak breakdown, so that many ranks stay fast)."""
    a = analyze(path, cfg, rank, composition=False)
    peaks = a.peaks.set_index("峰值")
    return {
        "rank": rank,
        "allocated 峰值": peaks.loc["allocated", "allocated"],
        "active 峰值": peaks.loc["active", "active"],
        "reserved 峰值": peaks.loc["reserved", "reserved"],
        "距设备": cfg["device"]["total_gib"] - peaks.loc["reserved", "reserved"],
        "active 峰值时刻": f"{peaks.loc['active', 't_s']:.1f}s {peaks.loc['active', 'L1']} {peaks.loc['active', 'L2']}",
        "release": a.totals["releases"],
        "unmap GiB": a.totals["unmap_gib"],
        "同时 pending 峰值": a.totals["pending_peak_gib"],
        "active 结束−开始": a.end["active"] - a.start["active"],
    }


def step_table(log_dir: Path) -> pd.DataFrame:
    """Per step, across ranks: step time and the peaks the trainer logs (torch.cuda max allocated / reserved)."""
    rows = []
    for log in sorted(log_dir.glob("rank*.log")):
        rank = int(re.search(r"rank(\d+)", log.name).group(1))
        for line in log.read_text(errors="replace").splitlines():
            if m := LOG_STEP.search(line):
                rows.append((rank, int(m.group(1)), *map(float, m.groups()[1:])))
    frame = pd.DataFrame(rows, columns=["rank", "step", "time", "max_memory", "reserved"])
    if frame.empty:
        return frame
    table = frame.groupby("step").agg(
        ranks=("rank", "nunique"),
        time_mean_s=("time", "mean"),
        time_max_s=("time", "max"),
        max_memory_mean=("max_memory", "mean"),
        max_memory_max=("max_memory", "max"),
        reserved_mean=("reserved", "mean"),
        reserved_max=("reserved", "max"),
        reserved_max_rank=("reserved", lambda s: int(frame.loc[s.idxmax(), "rank"])),
    )
    return table.reset_index()


def report(step_dir: Path, ranks: list[int], cfg: dict, log_dir: Path | None, workers: int) -> str:
    paths = [step_dir / f"rank{r}_memory_snapshot.pickle" for r in ranks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(rank_row, paths, ranks, [cfg] * len(ranks)))
    table = pd.DataFrame(rows)
    lines = [
        f"# 显存 profile：跨 rank（`{step_dir}`）",
        "",
        "## 1. 各 rank 的显存",
        "",
        _xychart(table),
        "",
        "柱：active 峰值；线：reserved 峰值（GiB）。",
        "",
        md_table(table),
        "",
        "读法：`release` 不为 0 的 rank 在这一步里有过分配器全量释放，host 会卡住，其余 rank 就在下一次集合通信里等它，"
        "所以只要有一个 rank 触顶，全局都会变慢。用 `mem_snapshot.py --rank r` 看这个 rank 是被什么请求触发、"
        "当时有多少 pending。`active 结束−开始` 明显大于 0 说明这一步有显存没有还回来。",
    ]
    step = re.search(r"step-(\d+)", str(step_dir))
    if log_dir and log_dir.is_dir() and not (steps := step_table(log_dir)).empty:
        profiled = int(step.group(1)) + 1 if step else None  # profiling_memory/step-N is the log's Step N + 1
        steps.insert(1, "profiled", ["←" if s == profiled else "" for s in steps.step])
        lines += [
            "",
            "## 2. 各 step 的显存（训练日志）",
            "",
            md_table(steps),
            "",
            "读法：`max_memory` / `reserved` 是 trainer 每步打印的 `torch.cuda` 峰值（GiB），不同 step 的数据不同，"
            "峰值也会不同。被采样的一步（←）应该和相邻的干净 step 接近；Step 1 含编译和缓存预热，一般不作参考。",
        ]
    return "\n".join(lines) + "\n"


def _xychart(table: pd.DataFrame) -> str:
    fmt = lambda s: ", ".join(f"{v:.1f}" for v in s)  # noqa: E731
    return "\n".join(
        [
            "```mermaid",
            "xychart-beta",
            '    title "active peak (bar) / reserved peak (line), GiB"',
            f"    x-axis [{', '.join(f'r{r}' for r in table['rank'])}]",
            '    y-axis "GiB"',
            f"    bar [{fmt(table['active 峰值'])}]",
            f"    line [{fmt(table['reserved 峰值'])}]",
            "```",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step_dir", type=Path, help="profiling_memory/step-N directory")
    parser.add_argument("--ranks", type=int, nargs="*", help="ranks to read, default: every snapshot in step_dir")
    parser.add_argument("--logs", type=Path, help="per-rank training logs, default: <run>/logs next to step_dir")
    parser.add_argument("--config", type=Path, help="rules file, defaults to profile/mem_config.toml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, help="write the Markdown report here instead of stdout")
    args = parser.parse_args()
    ranks = args.ranks or sorted(
        int(re.search(r"rank(\d+)_", p.name).group(1)) for p in args.step_dir.glob("rank*_memory_snapshot.pickle")
    )
    log_dir = args.logs or args.step_dir.resolve().parent.parent / "logs"
    text = report(args.step_dir, ranks, load_config(args.config), log_dir, args.workers)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
