#!/usr/bin/env python3
"""One rank, one training step: where the GPU memory goes, and why the caching allocator stalls.

Reads the snapshot written by ``PROFILE_MEMORY=1`` (``profiling_memory/step-N/rank{r}_memory_snapshot.pickle``: every
allocator event of the step with its Python stack, the segment table at the end, and the xtuner.L1 / L2 ranges) and
writes a Markdown report (tables and Mermaid charts):

1. Overview: the three counters at start, end and peak, headroom to the device, allocator releases.
2. The memory curve over the step and per L1 phase (forward.i, backward.i, clip, optimizer, ...).
3. What is alive at the active peak: by category, by the L2 phase that allocated it, and by allocation site.
4. Per L2 range (layer / module): what a forward leaves alive, the transient peak of recompute and backward.
5. Allocator pools, one per CUDA stream.
6. Pending frees: blocks freed on the host but held by ``record_stream`` until another stream's event completes.
7. ``release_cached_blocks`` episodes: which request ran out of memory, what was pending, how much was unmapped.
8. Large requests.

The counters are those of ``torch.cuda.memory_stats``: *allocated* drops when the host frees a tensor, *active* when the
block can be reused (after the events of ``record_stream`` completed), *reserved* is what the allocator has mapped.

Usage::

    python profile/mem_snapshot.py <rank*_memory_snapshot.pickle | profiling_memory/step-N> [--rank 0] \\
        [--out report.md] [--png curve.png]
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tomllib
from trace_extract import md_table


GIB = 1024**3
CONFIG_PATH = Path(__file__).with_name("mem_config.toml")
# action -> sign of its size in (allocated, active, reserved)
DELTA = {
    "alloc": (1, 1, 0),
    "free_requested": (-1, 0, 0),
    "free_completed": (0, -1, 0),
    "segment_alloc": (0, 0, 1),
    "segment_map": (0, 0, 1),
    "segment_free": (0, 0, -1),
    "segment_unmap": (0, 0, -1),
}
PHASES = ("forward", "recompute", "backward")
# Memory handed back to the driver: segment_unmap with expandable segments, segment_free without.  A run of them is one
# release_cached_blocks; outside of torch.cuda.empty_cache() it means an allocation found no block at the memory limit.
RETURN = ("segment_unmap", "segment_free")


@dataclass
class Analysis:
    """Everything the report shows; the DataFrames are already in GiB."""

    rank: int
    duration_s: float
    events: int
    start: dict[str, float]
    end: dict[str, float]
    peaks: pd.DataFrame
    curve: pd.DataFrame
    l1: pd.DataFrame
    l2: pd.DataFrame
    pools: pd.DataFrame
    pending: pd.DataFrame
    releases: pd.DataFrame
    large: pd.DataFrame
    composition: dict[str, pd.DataFrame] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)


def load_config(path: Path | None = None) -> dict:
    """Read ``mem_config.toml`` (or ``path``) and precompile its regexes."""
    cfg = tomllib.loads((path or CONFIG_PATH).read_text())
    cfg["l1"] = [(re.compile(r["match"]), r["label"]) for r in cfg["range"]["l1"]]
    cfg["l2"] = [(re.compile(r["match"]), r["label"]) for r in cfg["range"]["l2"]]
    cfg["skip"] = re.compile(cfg["site"]["skip"])
    cfg["caller_re"] = re.compile(cfg["site"]["caller"])
    cfg["pools"] = [(re.compile(r["match"]), r["pool"]) for r in cfg["pool"]["rules"]]
    cfg["categories"] = [(re.compile(r["match"]), r["category"]) for r in cfg["category"]["rules"]]
    return cfg


def find_snapshot(path: Path, rank: int) -> Path:
    """Accept a snapshot file or a ``profiling_memory/step-N`` directory."""
    return path if path.is_file() else path / f"rank{rank}_memory_snapshot.pickle"


def analyze(path: Path, cfg: dict, rank: int = 0, composition: bool = True) -> Analysis:
    """Replay one snapshot.  ``composition=False`` skips the second pass that breaks down the active peak."""
    snapshot = pickle.loads(path.read_bytes())
    device = max(range(len(snapshot["device_traces"])), key=lambda d: len(snapshot["device_traces"][d]))
    events = snapshot["device_traces"][device]
    segments = [s for s in snapshot["segments"] if s["device"] == device]
    sites = _Sites(events, cfg)
    marks = _range_marks(snapshot.get("external_annotations", []), device, cfg)

    # Counters at the end of the window come from the segment table; rewinding the events gives the start.
    end = defaultdict(lambda: np.zeros(3))
    for seg in segments:
        end[seg["stream"]][2] += seg["total_size"]
        for block in seg["blocks"]:
            if block["state"] in ("active_allocated", "active_pending_free"):
                end[seg["stream"]][1] += block["size"]
                end[seg["stream"]][0] += block["size"] * (block["state"] == "active_allocated")
    now = defaultdict(lambda: np.zeros(3), {s: v.copy() for s, v in end.items()})
    for e in events:
        if e["action"] in DELTA:
            now[e["stream"]] -= np.array(DELTA[e["action"]]) * e["size"]
    start = sum(now.values(), np.zeros(3))

    total = start.copy()
    t0 = events[0]["time_us"]
    curve, peaks = [], {}
    live: dict[int, tuple[int, str]] = {}  # addr -> (alloc event index, L2 label at allocation)
    alloc_labels: dict[int, tuple[str, str]] = {}  # alloc event index -> (L1, L2) at allocation
    pending: dict[int, tuple[float, int, int | None]] = {}  # addr -> (free time, size, alloc event index)
    pending_rows, release_rows, large_rows = [], [], []
    completed_run: list[tuple[int, int | None]] = []  # free_completed events since the last other event
    pool = defaultdict(lambda: {"alloc_bytes": 0, "allocs": 0, "reserved_peak": 0.0, "pending_peak": 0.0})
    pending_bytes: defaultdict[int, int] = defaultdict(int)  # stream -> bytes in ``pending``
    stream_sites: defaultdict[int, Counter] = defaultdict(Counter)
    instances: dict[int, dict] = {}
    open_ranges: dict[str, list[tuple[str, str, int]]] = {"l1": [], "l2": []}
    mark_i = 0
    active_peak_index = 0
    pending_max = 0.0

    for i, e in enumerate(events):
        t = e["time_us"]
        while mark_i < len(marks) and marks[mark_i][0] <= t:
            _apply_mark(marks[mark_i], open_ranges, instances, total, (t - t0) / 1e6)
            mark_i += 1
        l1 = open_ranges["l1"][-1] if open_ranges["l1"] else None
        l2 = open_ranges["l2"][-1] if open_ranges["l2"] else None
        action, size, stream, addr = e["action"], e["size"], e["stream"], e["addr"]
        if action in DELTA:
            delta = np.array(DELTA[action]) * size
            total += delta
            now[stream] += delta

        if action == "alloc":
            live[addr] = (i, l2[0] if l2 else "")
            alloc_labels[i] = (l1[0] if l1 else "", l2[0] if l2 else "")
            pool[stream]["allocs"] += 1
            pool[stream]["alloc_bytes"] += size
            stream_sites[stream][sites.match_text(i)] += size
            if size >= cfg["report"]["big_request_gib"] * GIB:
                large_rows.append((i, size, stream, l1[0] if l1 else "", l2[0] if l2 else ""))
        elif action == "free_requested":
            alloc_index = live.pop(addr, (None, ""))[0]
            nxt = events[i + 1] if i + 1 < len(events) else None
            # A plain free logs free_completed within microseconds; a block waiting for record_stream events completes
            # later, possibly with no other allocator event in between.
            if not (nxt and nxt["action"] == "free_completed" and nxt["addr"] == addr and nxt["time_us"] - t < 100):
                pending[addr] = (t, size, alloc_index)
                pending_bytes[stream] += size
        elif action == "free_completed" and addr in pending:
            t_free, _, alloc_index = pending.pop(addr)
            pending_bytes[stream] -= size
            pending_rows.append((alloc_index, size, stream, (t - t_free) / 1e3))
            completed_run.append((size, alloc_index))
        elif action in RETURN and events[i - 1]["action"] not in RETURN:
            release_rows.append(_release(events, i, completed_run, sites, l1, l2, t0))

        if action != "free_completed":
            completed_run = []
        for inst in (l1, l2):
            if inst:
                rec = instances[inst[2]]
                rec["active_max"] = max(rec["active_max"], total[1])
                rec["reserved_max"] = max(rec["reserved_max"], total[2])
                rec["releases"] += action in RETURN and events[i - 1]["action"] not in RETURN
        p = pool[stream]
        p["reserved_peak"] = max(p["reserved_peak"], now[stream][2])
        # From ``pending`` rather than active - allocated: every plain free passes through free_requested first.
        p["pending_peak"] = max(p["pending_peak"], pending_bytes[stream])
        for k, name in enumerate(("allocated", "active", "reserved")):
            if name not in peaks or total[k] > peaks[name][0]:
                peaks[name] = (total[k], i, total.copy(), l1[0] if l1 else "", l2[0] if l2 else "")
        active_peak_index = peaks["active"][1]
        pending_max = max(pending_max, sum(pending_bytes.values()))
        curve.append(((t - t0) / 1e6, *total))

    for mark in marks[mark_i:]:
        _apply_mark(mark, open_ranges, instances, total, (mark[0] - t0) / 1e6)
    for inst in instances.values():
        inst.setdefault("t1", (events[-1]["time_us"] - t0) / 1e6)
        inst.setdefault("alloc1", total[0])

    # The free memory of every pool before each release (and before the pending blocks its synchronization waited
    # for): tells a request that no pool could serve (memory held by other streams' pools) from one its own pool could
    # not fit (fragmentation), and both from memory that was only pending.
    free = _free_blocks(events, segments, {r["event"] for r in release_rows})
    for r in release_rows:
        pools = free[r["event"]]
        r["pool_cached_gib"], r["pool_largest_free_gib"] = (v / GIB for v in pools.get(r["stream"], (0, 0)))
        r["other_pools_cached_gib"] = sum(v[0] for s, v in pools.items() if s != r["stream"]) / GIB
    names = {s: _pool_name(s, stream_sites[s], cfg) for s in pool}
    result = Analysis(
        rank=rank,
        duration_s=(events[-1]["time_us"] - t0) / 1e6,
        events=len(events),
        start=dict(zip(("allocated", "active", "reserved"), start / GIB)),
        end=dict(zip(("allocated", "active", "reserved"), total / GIB)),
        peaks=_peak_table(peaks, events, sites, t0),
        curve=pd.DataFrame(curve, columns=["t_s", "allocated", "active", "reserved"]).assign(
            allocated=lambda f: f.allocated / GIB, active=lambda f: f.active / GIB, reserved=lambda f: f.reserved / GIB
        ),
        l1=_instance_table(instances, "l1"),
        l2=_l2_table(instances),
        pools=_pool_table(pool, names),
        pending=_pending_table(pending_rows, sites, names),
        releases=pd.DataFrame(release_rows).assign(pool=lambda f: f.get("stream", pd.Series(dtype=int)).map(names))
        if release_rows
        else pd.DataFrame(),
        large=_large_table(large_rows, sites, names),
    )
    result.totals = {
        "releases": len(release_rows),
        "unmap_gib": sum(e["size"] for e in events if e["action"] in RETURN) / GIB,
        "map_gib": sum(e["size"] for e in events if e["action"] in ("segment_map", "segment_alloc")) / GIB,
        "pending_frees": len(pending_rows),
        "pending_gib": sum(r[1] for r in pending_rows) / GIB,
        "pending_peak_gib": pending_max / GIB,
    }
    if composition:
        result.composition = _composition(
            events, active_peak_index, peaks["active"][2], sites, cfg, alloc_labels, names
        )
    return result


class _Sites:
    """Allocation site and project caller of an event, computed once per event on demand."""

    def __init__(self, events: list[dict], cfg: dict) -> None:
        self.events, self.cfg, self.cache = events, cfg, {}

    def get(self, index: int | None) -> tuple[str, str, str]:
        """(site, caller, text the category / pool rules match) of event ``index``."""
        if index is None:
            return "(allocated before step)", "", ""
        if index not in self.cache:
            frames = self.events[index].get("frames", [])
            site = next(
                (f for f in frames if not self.cfg["skip"].search(f"{f['filename']}:{f['name']}")),
                frames[0] if frames else None,
            )
            caller = next((f for f in frames if self.cfg["caller_re"].search(f["filename"])), None)
            self.cache[index] = (
                _short(site) or "(no Python frame)",
                _short(caller),
                "\n".join(f"{f['filename']}:{f['name']}" for f in (site, caller) if f),
            )
        return self.cache[index]

    def match_text(self, index: int) -> str:
        return self.get(index)[2]


def _short(frame: dict | None) -> str:
    return f"{Path(frame['filename']).name}:{frame['line']}:{frame['name']}" if frame else ""


def _label(rules: list[tuple[re.Pattern, str]], text: str) -> str | None:
    for pattern, label in rules:
        if m := pattern.search(text):
            return re.sub(r"\{(\d+)\}", lambda g: m.group(int(g.group(1))) or "", label)
    return None


def _phase(label: str) -> str:
    """``backward.3`` -> ``backward``: drop the micro-batch index of an L1 label."""
    return re.sub(r"\.\d+$", "", label)


def _range_marks(annotations: list[dict], device: int, cfg: dict) -> list[tuple[float, str, str, str, str]]:
    """(time, stage, level, label, raw name) of every L1 / L2 range boundary, in time order."""
    marks = []
    for a in annotations:
        if a.get("device", device) != device:
            continue
        for level in ("l1", "l2"):
            if (label := _label(cfg[level], a["name"])) is not None:
                marks.append((a["time_us"], a["stage"], level, label, a["name"]))
                break
    return sorted(marks, key=lambda m: (m[0], m[1] == "START"))


def _apply_mark(mark: tuple, open_ranges: dict, instances: dict, total: np.ndarray, t_s: float) -> None:
    _, stage, level, label, name = mark
    stack = open_ranges[level]
    if stage == "START":
        key = len(instances)
        instances[key] = {
            "level": level,
            "label": label,
            "t0": t_s,
            "alloc0": total[0],
            "active0": total[1],
            "active_max": total[1],
            "reserved_max": total[2],
            "releases": 0,
        }
        stack.append((label, name, key))
    else:
        for j in range(len(stack) - 1, -1, -1):
            if stack[j][1] == name:
                instances[stack[j][2]].update(t1=t_s, alloc1=total[0])
                del stack[j]
                break


def _release(events: list[dict], i: int, completed_run: list, sites: _Sites, l1, l2, t0: float) -> dict:
    """One release_cached_blocks episode starting at the first segment_unmap ``i``."""
    j = i
    unmapped = 0
    while j < len(events) and events[j]["action"] in RETURN:
        unmapped += events[j]["size"]
        j += 1
    request = next((k for k in range(j, len(events)) if events[k]["action"] == "alloc"), None)
    frames = events[i].get("frames", [])
    explicit = any(f["name"] == "empty_cache" for f in frames)
    by_source = Counter()
    for size, alloc_index in completed_run:
        by_source[sites.get(alloc_index)[0]] += size
    before = events[i - len(completed_run) - 1]["time_us"] if i - len(completed_run) > 0 else events[i]["time_us"]
    stream = events[request]["stream"] if request is not None else events[i]["stream"]
    return {
        "event": i - len(completed_run),
        "t_s": (events[i]["time_us"] - t0) / 1e6,
        "L1": l1[0] if l1 else "",
        "L2": l2[0] if l2 else "",
        "trigger": "torch.cuda.empty_cache()" if explicit else sites.get(request)[0] if request is not None else "",
        "request_gib": 0.0 if explicit or request is None else events[request]["size"] / GIB,
        "stream": stream,
        "pending_freed_gib": sum(by_source.values()) / GIB,
        "pending_sources": "; ".join(f"{s} {v / GIB:.2f}" for s, v in by_source.most_common(3)),
        "unmapped_gib": unmapped / GIB,
        "stall_ms": (events[request]["time_us"] - before) / 1e3 if request is not None else float("nan"),
    }


def _free_blocks(events: list[dict], segments: list[dict], at: set[int]) -> dict[int, dict[int, tuple[int, int]]]:
    """Per stream, (total free, largest free block) just before each event index in ``at``.

    Free blocks are the mapped ranges minus the live (allocated or pending) blocks; both are known exactly at the end
    of the window from the segment table and are rewound event by event.
    """
    mapped = [(s["address"], s["address"] + s["total_size"], s["stream"]) for s in segments]
    live = {}
    for seg in segments:
        addr = seg["address"]
        for block in seg["blocks"]:
            if block["state"] != "inactive":
                live[addr] = (addr + block["size"], seg["stream"])
            addr += block["size"]
    result = {}
    for i in range(len(events) - 1, min(at, default=len(events)) - 1, -1):
        e = events[i]
        begin, end, stream = e["addr"], e["addr"] + e["size"], e["stream"]
        if e["action"] == "alloc":
            live.pop(begin, None)
        elif e["action"] == "free_completed":
            live[begin] = (end, stream)
        elif e["action"] in ("segment_map", "segment_alloc"):
            mapped = [
                piece
                for lo, hi, s in mapped
                for piece in ((lo, min(hi, begin), s), (max(lo, end), hi, s))
                if piece[0] < piece[1]
            ]
        elif e["action"] in RETURN:
            mapped.append((begin, end, stream))
        if i in at:
            result[i] = _gaps(mapped, live)
    return result


def _gaps(mapped: list[tuple[int, int, int]], live: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    blocks = defaultdict(list)
    for begin, (end, stream) in live.items():
        blocks[stream].append((begin, end))
    result = {}
    for stream in {s for *_, s in mapped}:
        gaps = []
        ranges = []  # adjacent mapped chunks of an expandable segment form one range
        for lo, hi, _ in sorted(m for m in mapped if m[2] == stream):
            if ranges and lo <= ranges[-1][1]:
                ranges[-1][1] = max(ranges[-1][1], hi)
            else:
                ranges.append([lo, hi])
        used = sorted(blocks[stream])
        j = 0
        for lo, hi in ranges:
            cursor = lo
            while j < len(used) and used[j][0] < hi:
                gaps.append(used[j][0] - cursor)
                cursor = max(cursor, used[j][1])
                j += 1
            gaps.append(hi - cursor)
        gaps = [g for g in gaps if g > 0]
        result[stream] = (sum(gaps), max(gaps, default=0))
    return result


def _pool_name(stream: int, sites_by_bytes: Counter, cfg: dict) -> str:
    if stream == 0:
        return "compute"
    for text, _ in sites_by_bytes.most_common(3):
        if (name := _label([(p, n) for p, n in cfg["pools"]], text)) is not None:
            return name
    return f"stream {stream}"


def _peak_table(peaks: dict, events: list[dict], sites: _Sites, t0: float) -> pd.DataFrame:
    rows = []
    for name, (_, index, state, l1, l2) in peaks.items():
        site, caller, _ = sites.get(index)
        rows.append(
            {
                "峰值": name,
                "t_s": (events[index]["time_us"] - t0) / 1e6,
                "L1": l1,
                "L2": l2,
                "allocated": state[0] / GIB,
                "pending": (state[1] - state[0]) / GIB,
                "active": state[1] / GIB,
                "reserved": state[2] / GIB,
                "此刻的事件": f"{events[index]['action']} {site}",
                "caller": caller,
            }
        )
    return pd.DataFrame(rows)


def _instance_table(instances: dict, level: str) -> pd.DataFrame:
    rows = [
        {
            "L1": r["label"],
            "t0_s": r["t0"],
            "dur_s": r["t1"] - r["t0"],
            "allocated 开始": r["alloc0"] / GIB,
            "allocated 结束": r["alloc1"] / GIB,
            "active 峰值": r["active_max"] / GIB,
            "reserved 峰值": r["reserved_max"] / GIB,
            "release": r["releases"],
        }
        for r in instances.values()
        if r["level"] == level
    ]
    return pd.DataFrame(rows)


def _l2_table(instances: dict) -> pd.DataFrame:
    frame = pd.DataFrame([r for r in instances.values() if r["level"] == "l2"])
    if frame.empty:
        return frame
    frame["left"] = (frame.alloc1 - frame.alloc0) / GIB
    frame["transient"] = (frame.active_max - frame.active0) / GIB
    if frame.label.nunique() > 40:  # a deep model: one row per kind of range, e.g. layer*.backward
        frame["label"] = frame.label.str.replace(r"\d+", "*", regex=True)
    table = frame.groupby("label", sort=False).agg(
        n=("label", "size"),
        left=("left", "mean"),
        transient=("transient", "max"),
        active_max=("active_max", "max"),
        releases=("releases", "sum"),
    )
    table["active_max"] /= GIB
    return table.reset_index().rename(
        columns={
            "label": "L2",
            "n": "次数",
            "left": "结束时净增 allocated（均值）",
            "transient": "区间内瞬时增量（最大）",
            "active_max": "active 峰值",
            "releases": "release",
        }
    )


def _pool_table(pool: dict, names: dict) -> pd.DataFrame:
    rows = [
        {
            "pool": names[s],
            "stream": s,
            "分配次数": p["allocs"],
            "累计分配 GiB": p["alloc_bytes"] / GIB,
            "reserved 峰值": p["reserved_peak"] / GIB,
            "pending 峰值": p["pending_peak"] / GIB,
        }
        for s, p in pool.items()
    ]
    return pd.DataFrame(rows).sort_values("reserved 峰值", ascending=False)


def _pending_table(rows: list, sites: _Sites, names: dict) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["index", "size", "stream", "wait_ms"])
    frame["site"] = [sites.get(None if pd.isna(i) else int(i))[0] for i in frame["index"]]
    frame["caller"] = [sites.get(None if pd.isna(i) else int(i))[1] for i in frame["index"]]
    frame["pool"] = frame.stream.map(names)
    table = frame.groupby(["pool", "site", "caller"]).agg(
        次数=("size", "size"),
        单块_GiB=("size", "max"),
        累计_GiB=("size", "sum"),
        等待中位_ms=("wait_ms", "median"),
        等待最长_ms=("wait_ms", "max"),
    )
    table[["单块_GiB", "累计_GiB"]] /= GIB
    return table.reset_index().sort_values("累计_GiB", ascending=False)


def _large_table(rows: list, sites: _Sites, names: dict) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["index", "size", "stream", "l1", "l2"])
    frame["site"] = [sites.get(i)[0] for i in frame["index"]]
    frame["caller"] = [sites.get(i)[1] for i in frame["index"]]
    frame["pool"] = frame.stream.map(names)
    frame["phase"] = frame.l2.str.split(".").str[-1].where(frame.l2 != "", frame.l1.map(_phase))
    table = frame.groupby(["site", "caller", "pool"]).agg(
        次数=("size", "size"),
        最大_GiB=("size", "max"),
        累计_GiB=("size", "sum"),
        阶段=("phase", lambda s: ", ".join(dict.fromkeys(s))),
    )
    table[["最大_GiB", "累计_GiB"]] /= GIB
    return table.reset_index().sort_values("最大_GiB", ascending=False)


def _composition(
    events: list[dict],
    peak: int,
    state: np.ndarray,
    sites: _Sites,
    cfg: dict,
    labels: dict[int, tuple[str, str]],
    pools: dict[int, str],
) -> dict[str, pd.DataFrame]:
    """What is active at event ``peak``: blocks allocated in the window by category, L2 range and site."""
    live: dict[int, tuple[int, str, str]] = {}
    for i in range(peak + 1):
        e = events[i]
        if e["action"] == "alloc":
            live[e["addr"]] = (i, *labels[i])
        elif e["action"] == "free_completed":
            live.pop(e["addr"], None)
    rows = []
    for index, l1, l2 in live.values():
        site, caller, text = sites.get(index)
        category = _label(cfg["categories"], f"{text}\npool:{pools.get(events[index]['stream'], '')}")
        if category is None:
            phase = l2.split(".")[-1] if l2 else ""
            category = (
                f"activation ({phase})"
                if phase in PHASES
                else f"activation ({l2})"
                if l2
                else f"other ({_phase(l1) or '-'})"
            )
        rows.append((category, l2 or f"({_phase(l1)})", site, caller, events[index]["size"]))
    frame = pd.DataFrame(rows, columns=["category", "L2", "site", "caller", "size"])
    before = state[1] - frame["size"].sum()
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [["before step", "(before step)", "(allocated before step)", "", before]], columns=frame.columns
            ),
        ]
    )

    def group(keys: list[str]) -> pd.DataFrame:
        table = frame.groupby(keys).agg(GiB=("size", "sum"), 块数=("size", "size")).reset_index()
        table["GiB"] /= GIB
        small = table["GiB"] < cfg["report"]["min_gib"]
        if small.sum() > 1:
            rest = {k: "(其余)" for k in keys} | {"GiB": table.GiB[small].sum(), "块数": table["块数"][small].sum()}
            table = pd.concat([table[~small], pd.DataFrame([rest])])
        table["占比"] = table["GiB"] / (state[1] / GIB)
        return table.sort_values("GiB", ascending=False)

    return {"category": group(["category"]), "l2": group(["L2"]), "site": group(["category", "site", "caller"])}


def report(a: Analysis, cfg: dict, source: Path) -> str:
    """The Markdown report of one analysis."""
    total = cfg["device"]["total_gib"]
    top = cfg["report"]["top"]
    peaks = a.peaks.set_index("峰值")
    lines = [
        f"# 显存 profile：rank {a.rank}",
        "",
        f"数据：`{source}`，窗口 `{a.duration_s:.2f}s`、`{a.events:,}` 个分配器事件。",
        "",
        "## 1. 概览",
        "",
        md_table(
            pd.DataFrame(
                [
                    {"指标": name, "开始": a.start[name], "结束": a.end[name], "峰值": peaks.loc[name, name]}
                    for name in ("allocated", "active", "reserved")
                ]
            )
        ),
        "",
        md_table(
            pd.DataFrame(
                [
                    (
                        "reserved 峰值距设备总显存",
                        f"{total - peaks.loc['reserved', 'reserved']:.2f} GiB（设备 {total:g} GiB）",
                    ),
                    ("active 结束 − 开始（泄漏检查）", f"{a.end['active'] - a.start['active']:+.3f} GiB"),
                    ("release_cached_blocks", f"{a.totals['releases']} 次，unmap {a.totals['unmap_gib']:.1f} GiB"),
                    ("segment map（向驱动要显存）", f"{a.totals['map_gib']:.1f} GiB"),
                    (
                        "pending 释放",
                        f"{a.totals['pending_frees']} 次、累计 {a.totals['pending_gib']:.1f} GiB，"
                        f"同时 pending 最多 {a.totals['pending_peak_gib']:.2f} GiB",
                    ),
                ],
                columns=["项", "值"],
            )
        ),
        "",
        "读法：allocated 是活着的 tensor；active 再加上 host 已释放、但还在等其他 stream 事件的块（pending），"
        "只有 active 以外的部分能被复用；reserved 是分配器向驱动要到的显存。reserved 接近设备上限时，"
        "只要一次分配在池里找不到块，分配器就会走 `release_cached_blocks`（见第 7 节）。",
        "",
        md_table(a.peaks),
        "",
        "## 2. 一步内的显存曲线",
        "",
        _curve_chart(a.curve, total, cfg["report"]["chart_points"]),
        "",
        "图中三条线自上而下是 reserved、active、allocated（Mermaid xychart 没有图例）。各 L1 阶段：",
        "",
        md_table(a.l1),
        "",
        "读法：`forward.i` 结束时的 allocated 减去开始时的值，就是这个 micro-batch 为反向留下的激活；"
        "active 峰值通常在 `backward.i`，是留下的激活加上重算和反向的瞬时块。",
    ]
    if a.composition:
        comp = a.composition
        lines += [
            "",
            f"## 3. active 峰值时刻的构成（{peaks.loc['active', 'active']:.2f} GiB）",
            "",
            _pie("active 峰值构成（GiB）", comp["category"].set_index("category").GiB),
            "",
            md_table(comp["category"]),
            "",
            "按分配它的 L2 区间（前几项）：",
            "",
            md_table(comp["l2"].head(top)),
            "",
            "按调用点（前几项）：",
            "",
            md_table(comp["site"].head(top)),
            "",
            "读法：`before step` 是这一步开始前已经在的块（参数分片、优化器状态等常驻显存）；"
            "`activation (forward)` 是 checkpoint 保存的层输入，`activation (recompute)` / `(backward)` 是重算和反向的瞬时块；"
            "pending 块按分配时的类别计入。类别规则见 `mem_config.toml` 的 `[category]`。",
        ]
    if not a.l2.empty:
        lines += [
            "",
            "## 4. 各 L2 区间（层 / 模块）",
            "",
            md_table(a.l2.sort_values("active 峰值", ascending=False).head(2 * top)),
            "",
            "读法：`结束时净增 allocated` 是这个区间留下的显存，forward 为正表示为反向保存的激活，backward 为负表示释放；"
            "`区间内瞬时增量` 是区间内 active 相对开始时最多涨了多少，反映重算和反向的临时块。",
        ]
    lines += ["", "## 5. 分配池（每个 CUDA stream 一个）", "", md_table(a.pools)]
    lines += [
        "",
        "读法：缓存分配器按 stream 分池，一个池里的空闲块不能给另一个 stream 用。旁路 stream 的池越大，"
        "compute 能用的越少。",
        "",
        "## 6. pending 释放",
        "",
        md_table(a.pending.head(top)) if not a.pending.empty else "这一步没有 pending 释放。",
        "",
        "读法：`record_stream` 过的 tensor 被 host 释放后，要等它登记过的 stream 在 GPU 上执行到释放点才能复用，"
        "中间就是 pending。host 跑在 GPU 前面越远，等得越久；等待期间这块显存既不能用，也不会被计入可释放的缓存。",
        "",
        "## 7. release_cached_blocks",
        "",
    ]
    if a.releases.empty:
        lines.append("这一步没有发生 release。")
    else:
        cols = [
            "t_s",
            "L1",
            "L2",
            "trigger",
            "request_gib",
            "pool",
            "pool_cached_gib",
            "pool_largest_free_gib",
            "other_pools_cached_gib",
            "pending_freed_gib",
            "pending_sources",
            "unmapped_gib",
            "stall_ms",
        ]
        lines += [
            md_table(a.releases[cols]),
            "",
            "读法：`trigger` 是找不到块的那次分配，`pending_freed_gib` 是 release 开头同步等到的 pending 块（即当时被卡住、"
            "本来可以复用的显存），`unmapped_gib` 是随后被整体还给驱动、又要重新 map 的缓存。`stall_ms` 取上一个分配器事件"
            "到这次请求完成的间隔，是 host 卡顿的上界。`pool_cached_gib` / `other_pools_cached_gib` 是 release 前请求所在池"
            "和其他池里已缓存、未使用的显存，`pool_largest_free_gib` 是请求所在池里最大的一块空闲：池内空闲 ≥ request 而最大一块"
            "< request 是碎片，池内空闲小、其他池大说明空闲显存被"
            "其他 stream 的池占着；两者都小而 `pending_freed_gib` 大，就是 pending 块挡住了复用。",
        ]
    lines += ["", f"## 8. 大块分配（≥ {cfg['report']['big_request_gib']:g} GiB）", ""]
    lines.append(md_table(a.large.head(top)) if not a.large.empty else "没有这么大的分配。")
    return "\n".join(lines) + "\n"


def _curve_chart(curve: pd.DataFrame, total: float, points: int) -> str:
    bins = np.linspace(curve.t_s.iloc[0], curve.t_s.iloc[-1], points + 1)
    groups = curve.groupby(np.clip(np.digitize(curve.t_s, bins) - 1, 0, points - 1))
    series = groups[["allocated", "active", "reserved"]].max()
    fmt = lambda s: ", ".join(f"{v:.1f}" for v in s)  # noqa: E731
    return "\n".join(
        [
            "```mermaid",
            "xychart-beta",
            '    title "reserved / active / allocated (GiB)"',
            f'    x-axis "step time (s)" 0 --> {curve.t_s.iloc[-1]:.1f}',
            f'    y-axis "GiB" 0 --> {max(total, series.reserved.max()):.0f}',
            f"    line [{fmt(series.reserved)}]",
            f"    line [{fmt(series.active)}]",
            f"    line [{fmt(series.allocated)}]",
            "```",
        ]
    )


def _pie(title: str, values: pd.Series) -> str:
    return "\n".join(
        [
            "```mermaid",
            "pie showData",
            f"    title {title}",
            *(f'    "{k}" : {v:.2f}' for k, v in values.items() if v > 0),
            "```",
        ]
    )


def plot(a: Analysis, total: float, path: Path) -> None:
    """PNG of the three counters with the L1 phases shaded and the releases marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4.5))
    step = max(1, len(a.curve) // 20000)
    c = a.curve.iloc[::step]
    for name in ("reserved", "active", "allocated"):
        ax.plot(c.t_s, c[name], label=name, lw=1)
    ax.axhline(total, color="k", ls="--", lw=0.8, label=f"device {total:g} GiB")
    for row in a.l1.itertuples():
        if row.L1.startswith(("backward", "forward")):
            ax.axvspan(
                row.t0_s, row.t0_s + row.dur_s, color="C0" if row.L1.startswith("forward") else "C1", alpha=0.08
            )
    for t in a.releases.get("t_s", []):
        ax.axvline(t, color="r", lw=0.8, alpha=0.6)
    ax.set_xlabel("time in the profiled step (s); shaded: forward (blue) / backward (orange); red: release")
    ax.set_ylabel("GiB")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "snapshot", type=Path, help="rank*_memory_snapshot.pickle or a profiling_memory/step-N directory"
    )
    parser.add_argument("--rank", type=int, default=0, help="rank to read when given a directory")
    parser.add_argument("--config", type=Path, help="rules file, defaults to profile/mem_config.toml")
    parser.add_argument("--out", type=Path, help="write the Markdown report here instead of stdout")
    parser.add_argument("--png", type=Path, help="also write the memory curve as a PNG")
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = find_snapshot(args.snapshot, args.rank)
    result = analyze(path, cfg, args.rank)
    text = report(result, cfg, path)
    if args.out:
        args.out.write_text(text)
    else:
        print(text)
    if args.png:
        plot(result, cfg["device"]["total_gib"], args.png)


if __name__ == "__main__":
    main()
