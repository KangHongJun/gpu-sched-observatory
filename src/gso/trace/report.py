"""On-demand 캡처 SQLite에서 요약(JSON)과 텍스트 타임라인을 만든다. Textual을 쓰지 않는다.

모든 수치는 라벨이 붙은 추정/측정이며, 알 수 없는 값은 ``None``(JSON ``null``)이다.
컨텍스트 구간은 ``restore_save``와 ``hold`` 두 구성을 따로 보여 주고 합치지 않는다
(docs/verification/RESULTS.md 결정 2). GPU 번호는 이벤트 테이블의 ``gpuId`` = NVML 인덱스
(nsys-schema §2.4)이고, 커널의 ``deviceId``는 ``TARGET_INFO_CUDA_DEVICE``의
``(pid, cudaId) -> gpuId``로 호스트 GPU에 대응시킨다(§5.4). 대응을 알 수 없는 커널은
추측하지 않고 제외하며 notes에 적는다.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

from gso.analysis.intervals import (
    context_residency,
    interprocess_overlap,
    kernel_durations,
    launch_to_start_delays,
    restores_per_second,
    slice_durations,
    summarize_ns,
    switch_rate_by_gpu,
    switch_rate_is_lower_bound_by_gpu,
)
from gso.trace.nsys_sqlite import (
    capture_window,
    ctx_intervals,
    dropped_hints,
    load_ctx_events,
    load_kernels_with_stats,
    pid_of,
    raw_counts,
)
from gso.tui.widgets import NA, render_timeline

SCHEMA_VERSION = 1
CONSTRUCTIONS = ("restore_save", "hold")
TIMELINE_WIDTH = 100
ZOOM_CELLS = 100
ZOOM_CELL_NS = 50_000  # 100 cells x 50 us = 5 ms
UNMAPPED_MARK = "대응시킬 수 없어"  # in the note about kernels whose GPU is unknown
_RESTORE_START = 8

_BLOCK_HELP = {
    "restore_save": "estimate: each context's RESTORE_START -> SAVE_END intervals",
    "hold": "estimate: a context is assumed to hold the GPU from its RESTORE_START "
    "until another context's RESTORE_START",
    "kernels": "kernel start -> end intervals per process (overlap here is not SM-level "
    "simultaneous execution)",
}


def _kernel_gpu_map(path: Path) -> dict[tuple[int, int], int]:
    """``(pid, cudaId) -> gpuId`` from ``TARGET_INFO_CUDA_DEVICE``; empty if absent."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        try:
            rows = conn.execute("SELECT pid, cudaId, gpuId FROM TARGET_INFO_CUDA_DEVICE").fetchall()
        except sqlite3.OperationalError:
            return {}
    finally:
        conn.close()
    return {(p, c): g for p, c, g in rows if g is not None}


def _notes(mode: str, masked: bool, lower_bound: bool, masked_only: bool) -> list[str]:
    notes = [
        ("컨텍스트 구간은 restore_save와 hold 두 구성 모두 추정값이며, 데이터만으로는 "
         "어느 쪽이 실제 실행에 해당하는지 정할 수 없습니다."),
        "커널 구간의 겹침은 SM에서의 동시 실행과 같지 않습니다.",
    ]
    if masked:
        notes.append("root가 아니어서 다른 사용자 프로세스는 마스킹되어 PID를 알 수 없습니다.")
    if masked_only:
        notes.append("이 GPU의 컨텍스트 전환 이벤트는 모두 마스킹되어 전환율을 알 수 없습니다(N/A).")
    if lower_bound:
        notes.append(
            "컨텍스트 전환율은 하한입니다: 마스킹된 프로세스끼리의 전환은 세지 않습니다."
        )
    if mode == "global":
        notes.append("전역 캡처는 외부 프로세스의 CUDA 커널 구간을 기록하지 않습니다.")
    else:
        notes.append("컨텍스트 전환은 직접 실행한 프로세스 주변에서만 기록됩니다.")
    return notes


def build_report(
    sqlite_path: Path, *, gpu: int, mode: str, masked: bool
) -> tuple[dict, str]:
    """``(summary, timeline_text)`` for ``gpu``, loading the capture once."""
    path = Path(sqlite_path)
    w0, w1 = capture_window(path)
    window = (w0, w1)
    has_window = w1 > w0
    events = load_ctx_events(path, gpu_id=gpu)

    rate = switch_rate_by_gpu(events, w0, w1).get(gpu)
    lower = switch_rate_is_lower_bound_by_gpu(events, w0, w1).get(gpu, False)
    lower = bool(rate is not None and lower)
    masked_restores = sum(
        1 for e in events if e.tag == _RESTORE_START and e.pid is None and w0 <= e.ts_ns < w1
    )
    rps = {k: v for k, v in restores_per_second(events, w0, w1).items() if k[0] == gpu}

    ivs = {c: ctx_intervals(events, c, window=window) for c in CONSTRUCTIONS}
    keys = set(rps)
    for c in CONSTRUCTIONS:
        keys |= {k for k in ivs[c] if k[0] == gpu}
    per_context = []
    for key in sorted(keys, key=lambda k: (pid_of(k[1]) or 0, k[1])):
        entry: dict = {"pid": pid_of(key[1]), "restores_per_s": rps.get(key)}
        for c in CONSTRUCTIONS:
            cur = ivs[c].get(key, [])
            entry[c] = {
                "share": context_residency(cur, w0, w1),
                "slice_ms": summarize_ns(slice_durations(cur)),
            }
        per_context.append(entry)

    kernels, kstats = load_kernels_with_stats(path)
    gmap = _kernel_gpu_map(path)
    mine, unresolved, elsewhere = [], 0, 0
    for k in kernels:
        host = gmap.get((k.pid, k.device_id)) if k.pid is not None else None
        if host is None:
            unresolved += 1
        elif host == gpu:
            mine.append(k)
        else:
            elsewhere += 1
    by_pid: dict[int, list[tuple[int, int]]] = {}
    for k in mine:
        by_pid.setdefault(k.pid, []).append((k.start_ns, k.end_ns))
    counts = raw_counts(path)
    measured = mode != "global" and "CUPTI_ACTIVITY_KIND_KERNEL" in counts
    if measured:
        kernel = {
            "count": None if (unresolved and not mine) else len(mine),
            "duration_ms": summarize_ns(kernel_durations(mine)),
            "launch_to_start_ms": summarize_ns(launch_to_start_delays(mine)),
            "ambiguous_api_matches": kstats.ambiguous_api_matches,
            "pids": sorted(by_pid),
        }
    else:  # no kernel data was captured: unknown, not zero
        empty = {"n": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
        kernel = {
            "count": None,
            "duration_ms": dict(empty),
            "launch_to_start_ms": dict(empty),
            "ambiguous_api_matches": None,
            "pids": None,
        }
    overlap = interprocess_overlap(by_pid, w0, w1) if has_window else None

    notes = _notes(mode, masked, lower, rate is None and masked_restores > 0)
    if unresolved:
        notes.append(
            f"커널 {unresolved}개는 CUDA 장치를 호스트 GPU에 대응시킬 수 없어 "
            "(TARGET_INFO_CUDA_DEVICE 행 없음) 제외했습니다."
        )
    if elsewhere:
        notes.append(f"커널 {elsewhere}개는 다른 GPU에서 실행되어 제외했습니다.")

    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "gpu": gpu,
        "masked": masked,
        "window_s": (w1 - w0) / 1e9 if has_window else None,
        "ctx": {
            "switch_rate_per_s": rate,
            "switch_rate_is_lower_bound": lower,
            "masked_restore_events": masked_restores,
            "per_context": per_context,
        },
        "kernel": kernel,
        "interprocess_overlap": overlap,
        "dropped_hints": dropped_hints(path),
        "raw_counts": counts,
        "notes": notes,
    }
    blocks = []
    if mode == "global":
        by_c = {
            c: {pid_of(k[1]): v for k, v in ivs[c].items() if k[0] == gpu} for c in CONSTRUCTIONS
        }
        first = next(
            (e.ts_ns for e in events if e.tag == _RESTORE_START and e.pid is not None
             and w0 <= e.ts_ns < w1),
            None,
        )
        blocks += [(f"{c} zoom", by_c[c], first) for c in CONSTRUCTIONS]
        blocks += [(c, by_c[c], None) for c in CONSTRUCTIONS]
    else:
        first = min((s for ivl in by_pid.values() for s, _ in ivl if s < w1), default=None)
        blocks += [("kernels zoom", by_pid, first), ("kernels", by_pid, None)]
    return summary, _timeline_text(summary, window, blocks, masked_restores)


def build_summary(sqlite_path: Path, *, gpu: int, mode: str, masked: bool) -> dict:
    return build_report(sqlite_path, gpu=gpu, mode=mode, masked=masked)[0]


def _timeline_text(
    summary: dict, window: tuple[int, int], blocks, masked_restores: int,
    width: int = TIMELINE_WIDTH,
) -> str:
    """``blocks``: ``(name, by_pid, zoom_start)``; a zoom block covers
    ``ZOOM_CELLS x ZOOM_CELL_NS`` from ``zoom_start`` (None zoom_start = the whole window)."""
    w0, w1 = window
    win = summary["window_s"]
    head = (
        f"GPU {summary['gpu']} | mode: {summary['mode']} | "
        f"window: {NA if win is None else f'{win:.2f} s'} | "
        "on-demand capture of this window only (not a continuous view)"
    )
    lines = [head]
    if w1 > w0:
        cell_ms = (w1 - w0) / 1e6 / width
        lines.extend(textwrap.wrap(
            f"Overview: 1 cell = {cell_ms:.2f} ms; a cell is marked if the interval touches it at "
            "all, so alternation faster than a cell looks continuous (see the zoom blocks).",
            width=110,
        ))
    has_rows = any(any(k is not None for k in by) for _, by, _ in blocks)
    if masked_restores and has_rows:
        lines.append(f"masked contexts not shown: {masked_restores} restore events")
    for name, by_key, zoom_start in blocks:
        base = name.removesuffix(" zoom")
        lines.append("")
        zoom = name.endswith(" zoom")
        if zoom:
            title = f"[{name}] {_BLOCK_HELP[base]}"
            if zoom_start is None or not w0 <= zoom_start < w1:
                t0 = t1 = 0
                title += "; zoom: N/A (no start event before the capture end)"
            else:
                t0 = zoom_start
                t1 = min(t0 + ZOOM_CELLS * ZOOM_CELL_NS, w1)
                zoom_cells = max(1, (t1 - t0) // ZOOM_CELL_NS)
                title += (f"; zoom: +{(t0 - w0) / 1e9:.3f} s .. +{(t1 - t0) / 1e6:.3f} ms, "
                          f"1 cell = {ZOOM_CELL_NS // 1000} µs")
        else:
            t0, t1 = w0, w1
            title = f"[{name}] {_BLOCK_HELP[name]}; '=' = overlaps the cell"
        lines.extend(textwrap.wrap(title, width=110, subsequent_indent="    "))
        keys = sorted(k for k in by_key if k is not None)
        if not keys:
            if masked_restores:
                lines.append(f"  {NA} (intervals exist, but belong to other users' masked processes)")
            else:
                lines.append(f"  {NA} (no intervals for this GPU in the capture)")
            continue
        if zoom and t1 <= t0:
            lines.append(f"  {NA}")
            continue
        cells = zoom_cells if zoom else width
        if zoom:
            t1 = t0 + cells * ZOOM_CELL_NS  # whole cells inside the window (at least one)
        rows = render_timeline({k: by_key[k] for k in keys}, t0, t1, cells)
        for k in keys:
            lines.append(f"{k:>8} |{rows[k]}|")
        if zoom:
            end = f"+{cells * ZOOM_CELL_NS / 1e6:.3f} ms"
            lines.append(" " * 10 + "+0".ljust(cells - len(end)) + end)
        else:
            span = f"{(w1 - w0) / 1e9:.1f} s" if w1 > w0 else NA
            lines.append(" " * 10 + "0.0 s" + span.rjust(width - 5))
    return "\n".join(lines) + "\n"
