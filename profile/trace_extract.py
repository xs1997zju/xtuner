#!/usr/bin/env python3
"""Reduce XTuner torch-profiler traces to per-rank parquet tables, and load them back classified.

A time trace of a long-sequence step is ~1 GB of JSON per rank.  The analyses read two small tables instead:

* ``rank{r}.gpu.parquet``: every kernel / memcpy / memset with its stream, collective metadata, the CPU op that
  launched it (linked by ``External id``) with that op's input shapes, and the ``xtuner.L1`` / ``xtuner.L2`` /
  ``FSDP::`` ranges open when it was launched.
* ``rank{r}.host.parquet``: host-side waits (synchronize calls, ``.item()``) and allocator calls slower than 1 ms.

L1 ranges (training phases) live on the main thread while backward kernels are launched by the autograd thread, so
L1 is looked up by launch time on the main thread; L2 ranges (layers) are looked up on the launching thread.
Naming and grouping rules live in ``profile_config.toml`` and are applied by :func:`load_rank`, so editing them
takes effect without re-extracting.

Usage::

    python profile/trace_extract.py <profiling_time/step-N> <out_dir> [--workers 8]
"""

from __future__ import annotations

import argparse
import bisect
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import tomllib


CONFIG_PATH = Path(__file__).with_name("profile_config.toml")
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
RANGE_PREFIXES = ("xtuner.L1.", "xtuner.L2.", "Optimizer.step", "FSDP::")
SYNC_OPS = {"cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize", "aten::_local_scalar_dense"}
ALLOC_OPS = {"aten::empty", "aten::empty_strided"}
ALLOC_MIN_US = 1000.0
# Kernel args kept per GPU event; NCCL kernels carry the collective metadata.
GPU_ARGS = {
    "grid": "grid",
    "block": "block",
    "Collective name": "coll",
    "Process Group Description": "pg",
    "Process Group Ranks": "pg_ranks",
    "Group size": "pg_size",
    "In msg nelems": "in_nelems",
    "Out msg nelems": "out_nelems",
    "dtype": "dtype",
}


def extract(trace: Path, out_dir: Path) -> str:
    """Write ``rank{r}.gpu.parquet`` and ``rank{r}.host.parquet`` for one ``rank{r}_trace_time-*.json``."""
    rank = int(re.match(r"rank(\d+)_", trace.name).group(1))
    events = orjson.loads(trace.read_bytes())["traceEvents"]
    ranges: defaultdict[int, list[tuple[float, float, str]]] = defaultdict(list)
    ops: dict[int, tuple[str, str]] = {}
    launch: dict[int, tuple[int, float]] = {}
    gpu, host = [], []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, name, args = e.get("cat"), e.get("name", ""), e.get("args") or {}
        ts, dur = float(e["ts"]), float(e.get("dur") or 0.0)
        if cat in GPU_CATS:
            gpu.append(
                (
                    name,
                    ts,
                    dur,
                    int(args.get("stream", -1)),
                    args.get("correlation"),
                    args.get("External id"),
                    *(args.get(key) for key in GPU_ARGS),
                )
            )
        elif cat == "user_annotation" and name.startswith(RANGE_PREFIXES):
            ranges[e["tid"]].append((ts, ts + dur, name))
        elif cat == "cpu_op":
            ops[args.get("External id")] = (name, orjson.dumps(args.get("Input Dims", [])).decode())
            if name in SYNC_OPS or (name in ALLOC_OPS and dur >= ALLOC_MIN_US):
                host.append((name, ts, dur, e["tid"], ts))
        elif cat in ("cuda_runtime", "cuda_driver"):
            if args.get("correlation") is not None:
                launch[args["correlation"]] = (e["tid"], ts)
            if name in SYNC_OPS:
                host.append((name, ts, dur, e["tid"], ts))
    del events

    columns = ["name", "ts", "dur", "stream", "correlation", "op_id", *GPU_ARGS.values()]
    gpu_frame = pd.DataFrame(gpu, columns=columns).astype({"grid": str, "block": str, "pg_ranks": str})
    op = [ops.get(i, ("", "[]")) for i in gpu_frame.op_id]
    gpu_frame["op"], gpu_frame["op_dims"] = [o[0] for o in op], [o[1] for o in op]
    where = [launch.get(c, (-1, np.nan)) for c in gpu_frame["correlation"]]
    gpu_frame["tid"], gpu_frame["launch_ts"] = [w[0] for w in where], [w[1] for w in where]
    host_frame = pd.DataFrame(host, columns=["name", "ts", "dur", "tid", "launch_ts"])
    for frame in (gpu_frame, host_frame):
        _add_ranges(frame, ranges)

    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_frame.drop(columns=["correlation"]).to_parquet(out_dir / f"rank{rank}.gpu.parquet", index=False)
    host_frame.to_parquet(out_dir / f"rank{rank}.host.parquet", index=False)
    return f"rank{rank}: {len(gpu_frame)} GPU events, {int((gpu_frame.l2 != '').sum())} inside an L2 range"


def load_config(path: Path | None = None) -> dict:
    """Read ``profile_config.toml`` (or ``path``) and precompile its regexes."""
    cfg = tomllib.loads((path or CONFIG_PATH).read_text())
    cfg["l4"] = [(re.compile(r["match"]), r["label"]) for r in cfg["l4"]["rules"]]
    cfg["l3"] = [(re.compile(r["match"]), r["block"]) for r in cfg["l3"]["rules"]]
    cfg["flops"] = [(re.compile(r["op"]), r) for r in cfg["flops"]["rules"]]
    return cfg


def load_rank(pq_dir: Path, rank: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the classified GPU events and the host events of one rank.

    GPU columns added here: ``end``, ``resource`` (compute / deepep / nccl / memcpy), ``group`` / ``verb`` for NCCL,
    ``l4``, ``l3``, ``phase``, ``flops`` (on the longest kernel of each op instance, so sums count it once), ``mnk``
    (GEMM shape) and ``peak_tflops``.
    """
    gpu = pd.read_parquet(pq_dir / f"rank{rank}.gpu.parquet")
    host = pd.read_parquet(pq_dir / f"rank{rank}.host.parquet")
    gpu["end"] = gpu.ts + gpu.dur
    res = cfg["resource"]
    gpu["resource"] = np.select(
        [gpu.name.str.contains(res[r]) for r in ("nccl", "deepep", "memcpy")], ["nccl", "deepep", "memcpy"], "compute"
    )
    gpu["group"] = gpu.pg.map(cfg["comm"]["groups"]).fillna(gpu.pg).fillna("")
    gpu["verb"] = gpu.coll.map(cfg["comm"]["verbs"]).fillna(gpu.coll).fillna("")

    keys = list(zip(gpu.op, gpu.name, gpu.resource, gpu.group, gpu.verb))
    labels = {key: _classify(*key, cfg) for key in set(keys)}
    gpu["l4"] = [labels[k][0] for k in keys]
    gpu["l3"] = [labels[k][1] for k in keys]
    gpu["phase"] = [phase_of(l1, l2) for l1, l2 in zip(gpu.l1, gpu.l2)]

    info = {key: _flops(*key, cfg) for key in set(zip(gpu.op, gpu.op_dims))}
    per_kernel = [info[key] for key in zip(gpu.op, gpu.op_dims)]
    # FLOPs belong to the op instance: put them on its longest kernel (not a memset it launched first), so sums
    # over kernels count each op once and TFLOP/s divides by the kernel that did the math.
    main = np.zeros(len(gpu), dtype=bool)
    main[gpu[gpu.op_id.notna()].groupby("op_id").dur.idxmax().to_numpy()] = True
    gpu["flops"] = np.where(main, [f[0] for f in per_kernel], np.nan)
    gpu["mnk"] = [f[1] for f in per_kernel]
    gpu["peak_tflops"] = [f[2] for f in per_kernel]
    return gpu, host


def phase_of(l1: str, l2: str) -> str:
    """``forward`` / ``recompute`` / ``backward`` from the L2 range, else the L1 phase (``forward.3`` -> ``forward``)."""
    for phase in ("recompute", "backward", "forward"):
        if l2.endswith(phase):
            return phase
    if l1.startswith("Optimizer.step"):
        return "optimizer"
    return re.sub(r"\.\d+$", "", l1.removeprefix("xtuner.L1.")) or "outside L1"


def union_s(start: np.ndarray, end: np.ndarray) -> float:
    """Length of the union of ``[start, end)`` intervals, microseconds in, seconds out."""
    s, e = merge(start, end)
    return float((e - s).sum()) / 1e6


def merge(start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge ``[start, end)`` intervals into sorted, disjoint ones."""
    if len(start) == 0:
        return np.array([]), np.array([])
    order = np.argsort(start)
    start, end = np.asarray(start)[order], np.maximum.accumulate(np.asarray(end)[order])
    head = np.r_[True, start[1:] > end[:-1]]
    return start[head], end[np.r_[head[1:], True]]


def md_table(frame: pd.DataFrame, digits: int = 2) -> str:
    """Render a DataFrame as a Markdown table; floats get ``digits`` decimals."""

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:,.{digits}f}"
        return str(value).replace("|", "\\|")

    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    align = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(t) else "---" for t in frame.dtypes) + "|"
    return "\n".join(
        [head, align, *("| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False))]
    )


def _add_ranges(frame: pd.DataFrame, ranges: dict[int, list[tuple[float, float, str]]]) -> None:
    """Add ``l1`` (main thread, by launch time) and ``l2`` / ``l2_outer`` / ``l2_id`` / ``fsdp`` (launching thread)."""
    main = next((tid for tid, spans in ranges.items() if any(n.startswith("xtuner.L1.") for *_, n in spans)), None)
    level1 = sorted(s for s in ranges.get(main, []) if s[2].startswith(("xtuner.L1.", "Optimizer.step")))
    starts = [s[0] for s in level1]
    l1 = []
    for ts in frame.launch_ts:
        i = bisect.bisect_right(starts, ts) - 1
        l1.append(level1[i][2] if i >= 0 and level1[i][1] >= ts else "")
    frame["l1"] = l1

    n = len(frame)
    l2, l2_outer, l2_id, fsdp = [""] * n, [""] * n, [np.nan] * n, [""] * n
    launch_ts = frame.launch_ts.to_numpy()
    for tid, rows in frame.groupby("tid").indices.items():
        spans = ranges.get(tid, [])
        # Sweep: range starts sort before launches, launches before range ends at equal timestamps.
        marks = [(s, 0, k) for k, (s, _, _) in enumerate(spans)] + [(e, 2, k) for k, (_, e, _) in enumerate(spans)]
        marks += [(launch_ts[i], 1, i) for i in rows]
        marks.sort(key=lambda m: (m[0], m[1]))
        active: dict[int, tuple[float, str]] = {}
        for _, kind, k in marks:
            if kind == 0:
                active[k] = spans[k][0], spans[k][2]
            elif kind == 2:
                active.pop(k, None)
            else:
                layers = sorted(v for v in active.values() if v[1].startswith("xtuner.L2."))
                if layers:
                    l2[k], l2_outer[k], l2_id[k] = layers[-1][1], layers[0][1], layers[-1][0]
                modules = [v for v in active.values() if v[1].startswith("FSDP::")]
                if modules:
                    fsdp[k] = max(modules)[1]
    frame["l2"], frame["l2_outer"], frame["l2_id"], frame["fsdp"] = l2, l2_outer, l2_id, fsdp


def _classify(op: str, kernel: str, resource: str, group: str, verb: str, cfg: dict) -> tuple[str, str]:
    """``(l4, l3)`` of one (op, kernel) pair."""
    if resource == "nccl":
        return f"{group} {verb}".strip(), f"{group} comm"
    key = f"{op}|{kernel}"
    label = op or kernel[:48]
    for pattern, template in cfg["l4"]:
        match = pattern.search(key)
        if match:
            label = template.format("", *(g or "" for g in match.groups()), op=op, kernel=kernel)[:48]
            break
    if resource == "memcpy":
        return label, "memcpy"
    key = f"{label}|{key}"
    return label, next((block for pattern, block in cfg["l3"] if pattern.search(key)), "elementwise")


def _flops(op: str, dims: str, cfg: dict) -> tuple[float, str, float]:
    """``(flops, "MxNxK", peak TFLOP/s)`` of one op instance from its input dims; NaN when no rule applies."""
    for pattern, rule in cfg["flops"]:
        if not pattern.search(op):
            continue
        peak = cfg["device"]["peak_tflops"][rule["dtype"]]
        try:
            scope = {"d": orjson.loads(dims)}
            if "mnk" in rule:
                m, n, k = eval(rule["mnk"], {"__builtins__": {}}, scope)
                return 2.0 * m * n * k, f"{m}x{n}x{k}", peak
            return float(eval(rule["flops"], {"__builtins__": {}}, scope)), "", peak
        except (IndexError, TypeError, ValueError):
            return np.nan, "", np.nan
    return np.nan, "", np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", type=Path, help="profiling_time/step-N directory with rank*_trace_time-*.json")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    traces = sorted(args.trace_dir.glob("rank*_trace*.json"))
    if not traces:
        parser.error(f"no rank*_trace*.json under {args.trace_dir}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(extract, traces, [args.out_dir] * len(traces)):
            print(line, flush=True)


if __name__ == "__main__":
    main()
