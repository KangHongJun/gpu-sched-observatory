#!/usr/bin/env python3
"""Derive restore_save / hold interval totals from the raw exports (read-only).

Window (single definition): w0 = 0; w1 = max(ts of the "Profiling has stopped." DIAGNOSTIC_EVENT row,
largest GPU_CONTEXT_SWITCH_EVENTS.timestamp over all gpuIds incl. masked rows; largest event ts if no diag row).
"""

import sqlite3
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "docs" / "verification" / "raw"
MASKED = 1 << 48


def ro(n):
    return sqlite3.connect(f"file:{RAW / n}?mode=ro", uri=True)


def pct(v, p):
    v = sorted(v)
    return v[round(p / 100 * (len(v) - 1))]


def build(rows, w0, w1):
    """rows: (tag, globalPid, ts) of one gpuId in (ts, seqNo) order. Returns (restore_save, hold, counts)."""
    rs, op, seen = [], {}, set()
    c = {
        "unmatched7": 0,
        "reopen8": 0,
        "leading7": 0,
        "trailing8": 0,
        "hold_same8": 0,
        "hold_leading7": 0,
    }
    for t, g, ts in rows:
        if g == MASKED:
            continue
        if t == 8:
            if g in op:
                c["reopen8"] += 1
            op[g] = ts
        elif t == 7:
            if g in op:
                rs.append((g, op.pop(g), ts))
            elif g not in seen:
                rs.append((g, w0, ts))
                c["leading7"] += 1
            else:
                c["unmatched7"] += 1
        seen.add(g)
    for g, t0 in op.items():
        rs.append((g, t0, w1))
        c["trailing8"] += 1
    hold, cur, seen_h = [], None, set()
    for t, g, ts in rows:
        if t == 7 and g != MASKED and g not in seen_h and cur is None:
            # choice: first event of a context is a 7 and nobody holds yet -> it held since w0
            cur = (g, w0)
            c["hold_leading7"] += 1
        if g != MASKED:
            seen_h.add(g)
        if t != 8:
            continue
        if cur is not None and g == cur[0]:
            c["hold_same8"] += 1
            continue
        if cur is not None and cur[0] != MASKED:
            hold.append((cur[0], cur[1], ts))
        cur = (g, ts)
    if cur is not None and cur[0] != MASKED:
        hold.append((cur[0], cur[1], w1))
    return rs, hold, c


def report(label, rows, w0, w1, name):
    rs, hold, c = build(rows, w0, w1)
    span = w1 - w0
    print(f"--- {label}: window [{w0}, {w1}] span={span / 1e9:.6f} s, counts={c}")
    for nm, iv in (("restore_save", rs), ("hold", hold)):
        tot = {}
        for g, a, b in iv:
            tot[g] = tot.get(g, 0) + b - a
        parts = ", ".join(
            f"{name(g)}={v / 1e9:.3f} s ({100 * v / span:.1f}%)"
            for g, v in sorted(tot.items())
        )
        sm = sum(tot.values())
        iv_sorted = sorted(iv, key=lambda x: x[1])
        ov = sum(
            1
            for i in range(len(iv_sorted) - 1)
            if iv_sorted[i + 1][1] < iv_sorted[i][2]
            and iv_sorted[i + 1][0] != iv_sorted[i][0]
        )
        print(
            f"  {nm}: n={len(iv)} {parts}; combined={sm / 1e9:.3f} s ({100 * sm / span:.1f}%); cross-ctx overlaps={ov}"
        )
    return rs, hold


def main():
    import json

    a = json.loads((RAW / "04_wl_a.json").read_text())["pid"]
    b = json.loads((RAW / "04_wl_b.json").read_text())["pid"]
    dec = lambda g: (g >> 24) & 0xFFFFFF
    nm = lambda g: "A" if dec(g) == a else "B" if dec(g) == b else "P"
    con = ro("ctx_nonroot.sqlite")
    rows = con.execute(
        "select tag, globalPid, timestamp from GPU_CONTEXT_SWITCH_EVENTS where gpuId=1 order by timestamp, seqNo"
    ).fetchall()
    stop0 = con.execute(
        "select max(timestamp) from DIAGNOSTIC_EVENT where text='Profiling has stopped.'"
    ).fetchone()[0]
    maxev = con.execute(
        "select max(timestamp) from GPU_CONTEXT_SWITCH_EVENTS"
    ).fetchone()[0]
    w0, w1 = 0, max(stop0, maxev)
    print(
        f"window: w0=0, stop diag={stop0}, max event ts (all gpuIds)={maxev}, w1={w1}"
    )
    rs, _hold = report("ctx_nonroot GPU 1", rows, w0, w1, nm)
    # same-context 8->7 (restore_save) and cross-context 7->8 distributions
    d = [y - x for _, x, y in rs]
    print(
        f"  restore_save length us p1/p50/p99: {pct(d, 1) / 1e3:.1f}/{pct(d, 50) / 1e3:.3f}/{pct(d, 99) / 1e3:.1f}"
    )
    cross = []
    last7 = None
    for t, g, ts in rows:
        if t == 7:
            last7 = (g, ts)
        elif t == 8 and last7 and last7[0] != g:
            cross.append(ts - last7[1])
            last7 = None
    print(
        f"  cross-context 7->8 n={len(cross)} p1/p50/p99 us: {pct(cross, 1) / 1e3:.1f}/{pct(cross, 50) / 1e3:.1f}/{pct(cross, 99) / 1e3:.1f} sum={sum(cross) / 1e9:.3f} s"
    )
    # launched capture, GPU 1
    lc = ro("ctx_launched.sqlite")
    lrows = lc.execute(
        "select tag, globalPid, timestamp from GPU_CONTEXT_SWITCH_EVENTS where gpuId=1 order by timestamp, seqNo"
    ).fetchall()
    stop = lc.execute(
        "select timestamp from DIAGNOSTIC_EVENT where text='Profiling has stopped.'"
    ).fetchone()[0]
    kmin, kmax = lc.execute(
        "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    print(
        f"launched GPU1 events={len(lrows)} kernels start {kmin / 1e9:.3f}..{kmax / 1e9:.3f} s, stop diag {stop / 1e9:.3f} s"
    )
    lmax = lc.execute(
        "select max(timestamp) from GPU_CONTEXT_SWITCH_EVENTS"
    ).fetchone()[0]
    lw1 = max(stop, lmax)
    report(f"launched GPU 1, window [0, {lw1}]", lrows, 0, lw1, lambda g: "P")
    closed = [(lrows[0][2], lrows[1][2])]
    n = lc.execute(
        "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?",
        closed[0],
    ).fetchone()[0]
    print(
        f"kernels overlapping the one closed 8->7 pair {closed[0]}: {n}; first kernel start {kmin} > last 8 {lrows[-1][2]}: {kmin > lrows[-1][2]}"
    )
    names = dict(lc.execute("select id, value from StringIds"))
    for fn in ("cudaDeviceSynchronize", "cudaLaunchKernel_v7000"):
        ids = [i for i, v in names.items() if v.startswith(fn)]
        k = 0
        for i in ids:
            k += lc.execute(
                "select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where nameId=?", (i,)
            ).fetchone()[0]
        print(f"runtime calls {fn}*: {k}")
    print(
        "kernels:",
        lc.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0],
    )


if __name__ == "__main__":
    sys.exit(main())
