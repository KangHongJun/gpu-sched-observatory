#!/usr/bin/env python3
"""Cross-check gso's loader/metrics against independent SQL over a raw nsys export (read-only).

Usage: uv run python scripts/verify/21_crosscheck.py <capture.sqlite> --gpu N

The window-rule comparison restates the documented rule (weak evidence) and the raw_counts
comparison is a low-value consistency check; the counts, sums and first/last values are the
substantive ones.

Every comparison puts a number computed here with plain SQL (written from
docs/verification/nsys-schema.md, section 2-5 and 10) next to the number gso reports.
Exit 0 if all match, 1 otherwise. PIDs are never printed (processes are labelled A, B, ...
in order of first appearance; masked rows are "masked"). PROCESSES is never queried.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

from gso.analysis.intervals import kernel_durations, restores_per_second
from gso.trace.nsys_sqlite import (
    capture_window,
    load_ctx_events,
    load_kernels,
)
from gso.trace.report import build_summary

MASKED = 1 << 48  # schema 2.3
PID = "((globalPid >> 24) & 16777215)"  # schema 2.2
READ_TABLES = (
    "GPU_CONTEXT_SWITCH_EVENTS",
    "ENUM_GPU_CTX_SWITCH",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "StringIds",
    "DIAGNOSTIC_EVENT",
)


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _tables(c: sqlite3.Connection) -> set[str]:
    return {
        r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return (
            a is not None
            and b is not None
            and math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
        )
    return a == b


def run(path: Path, gpu: int) -> list[tuple[str, object, object, bool]]:
    rows: list[tuple[str, object, object]] = []
    with _ro(path) as c:
        tabs = _tables(c)
        has_ctx = "GPU_CONTEXT_SWITCH_EVENTS" in tabs
        has_k = "CUPTI_ACTIVITY_KIND_KERNEL" in tabs

        # independent window: w0 = 0, w1 = max(stopped row, max event ts over all gpus)
        stopped = max_ev = None
        if "DIAGNOSTIC_EVENT" in tabs:
            stopped = c.execute(
                "SELECT max(timestamp) FROM DIAGNOSTIC_EVENT WHERE text = 'Profiling has stopped.'"
            ).fetchone()[0]
        if has_ctx:
            max_ev = c.execute(
                "SELECT max(timestamp) FROM GPU_CONTEXT_SWITCH_EVENTS"
            ).fetchone()[0]
        cand = [v for v in (stopped, max_ev) if v is not None]
        w1 = max(cand) if cand else 0

        # label processes by first appearance (events, then kernels)
        labels: dict[int, str] = {}

        def label(pid: int) -> str:
            if pid not in labels:
                labels[pid] = (
                    chr(ord("A") + len(labels))
                    if len(labels) < 26
                    else f"P{len(labels)}"
                )
            return labels[pid]

        if has_ctx:
            for (pid,) in c.execute(
                f"SELECT {PID} FROM GPU_CONTEXT_SWITCH_EVENTS WHERE gpuId = ? AND globalPid != ?"
                " GROUP BY 1 ORDER BY min(timestamp)",
                (gpu, MASKED),
            ):
                label(pid)
        if has_k:
            for (pid,) in c.execute(
                f"SELECT {PID} FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1 ORDER BY min(start)"
            ):
                label(pid)

        events = load_ctx_events(path)
        kernels = load_kernels(path)

        if has_ctx:
            n = c.execute("SELECT COUNT(*) FROM GPU_CONTEXT_SWITCH_EVENTS").fetchone()[
                0
            ]
            rows.append(("ctx events: count (all GPUs)", n, len(events)))
        if has_k:
            n = c.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[
                0
            ]
            rows.append(("kernels: count", n, len(kernels)))
            s = c.execute(
                "SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
            ).fetchone()[0]
            rows.append(
                ("kernels: sum(end - start) ns", s or 0, sum(kernel_durations(kernels)))
            )
            for pid, cnt, first, last in c.execute(
                f"SELECT {PID}, COUNT(*), MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
                " GROUP BY 1 ORDER BY min(start)"
            ):
                mine = [k for k in kernels if k.pid == pid]
                nm = f"kernels, process {label(pid)}"
                rows.append((f"{nm}: count", cnt, len(mine)))
                rows.append(
                    (
                        f"{nm}: first start ns",
                        first,
                        min((k.start_ns for k in mine), default=None),
                    )
                )
                rows.append(
                    (
                        f"{nm}: last end ns",
                        last,
                        max((k.end_ns for k in mine), default=None),
                    )
                )

        if has_ctx:
            sql = {}
            for g, pid, masked, tag, cnt in c.execute(
                f"SELECT gpuId, {PID}, globalPid = ?, tag, COUNT(*) FROM GPU_CONTEXT_SWITCH_EVENTS"
                " WHERE gpuId = ? AND tag IN (7, 8) GROUP BY 1, 2, 3, 4",
                (MASKED, gpu),
            ):
                sql[(None if masked else pid, tag)] = cnt
            keys = sorted(
                {k[0] for k in sql}, key=lambda p: (p is None, labels.get(p, ""))
            )
            # SQL's own 'first seen' ordering is by label; masked last
            for pid in keys:
                nm = "masked" if pid is None else f"process {label(pid)}"
                for tag, tname in ((8, "RESTORE_START"), (7, "SAVE_END")):
                    got = sum(
                        1
                        for e in events
                        if e.gpu_id == gpu
                        and e.tag == tag
                        and e.pid == pid
                        and (pid is not None or e.global_pid == MASKED)
                    )
                    rows.append(
                        (
                            f"GPU {gpu} ctx, {nm}: {tname} (tag {tag}) count",
                            sql.get((pid, tag), 0),
                            got,
                        )
                    )

            # Rate numerator: tag 8 with 0 <= timestamp < w1. The half-open end follows the
            # [t0, t1) definition in gso.analysis.intervals (an event exactly at w1 is not counted).
            bounded = {
                pid: n
                for pid, n in c.execute(
                    f"SELECT {PID}, COUNT(*) FROM GPU_CONTEXT_SWITCH_EVENTS WHERE gpuId = ?"
                    " AND tag = 8 AND globalPid != ? AND timestamp >= 0 AND timestamp < ?"
                    " GROUP BY 1",
                    (gpu, MASKED, w1),
                )
            }
            secs = w1 / 1e9
            gw0, gw1 = capture_window(path)
            rps = restores_per_second(events, gw0, gw1) if gw1 > gw0 else {}
            for pid in keys:
                if pid is None:
                    continue
                exp = bounded.get(pid, 0) / secs if secs > 0 else None
                got = next(
                    (
                        v
                        for (g, gp), v in rps.items()
                        if g == gpu and (gp >> 24) & 0xFFFFFF == pid
                    ),
                    None,
                )
                rows.append(
                    (
                        f"GPU {gpu} restores/s, process {label(pid)} (window {secs:.3f} s)",
                        exp,
                        got,
                    )
                )
            rows.append(("capture window w1 ns", w1, capture_window(path)[1]))

        summ = build_summary(
            path, gpu=gpu, mode="launched" if has_k else "global", masked=True
        )
        for t in sorted(tabs & set(READ_TABLES), key=READ_TABLES.index):
            n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            rows.append((f"report raw_counts: {t}", n, summ["raw_counts"].get(t)))
        for t in summ["raw_counts"]:
            if t not in tabs:
                rows.append(
                    (
                        f"report raw_counts: {t} (not in file)",
                        None,
                        summ["raw_counts"][t],
                    )
                )
    return [(n, a, b, _same(a, b)) for n, a, b in rows]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("capture", type=Path)
    ap.add_argument("--gpu", type=int, required=True)
    args = ap.parse_args(argv)
    res = run(args.capture, args.gpu)
    w = max((len(r[0]) for r in res), default=10)
    print(f"{'comparison'.ljust(w)} | {'SQL':>22} | {'gso':>22} | match")
    print(f"{'-' * w}-+-{'-' * 22}-+-{'-' * 22}-+------")
    for name, a, b, ok in res:
        fa = f"{a:.6f}" if isinstance(a, float) else str(a)
        fb = f"{b:.6f}" if isinstance(b, float) else str(b)
        print(f"{name.ljust(w)} | {fa:>22} | {fb:>22} | {'✓' if ok else '✗'}")
    bad = sum(1 for r in res if not r[3])
    print(f"\n{len(res) - bad}/{len(res)} comparisons match")
    return 0 if res and bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
