"""Pure formatting helpers for the live screen (no Textual import).

Rule: an unknown value is "N/A", never 0 and never a stale number shown as current.
"""

from __future__ import annotations

from dataclasses import dataclass

from gso.record import Record

NA = "N/A"
GAP = "·"
_BLOCKS = "▁▂▃▄▅▆▇█"


def bar(value: float | None, width: int = 16) -> str:
    """``[####----] 78%``. The fill is the value clamped to 0-100 rounded to the nearest cell (Python round, half to even);
    the number shown is the rounded (unclamped) value. None -> centred N/A, same total length."""
    if value is None:
        return f"[{NA.center(width)}] {NA}"
    frac = min(max(value, 0.0), 100.0) / 100.0
    fill = round(frac * width)
    return f"[{'#' * fill}{'-' * (width - fill)}] {value:.0f}%"


def latest(
    records: list[Record],
    metric: str,
    pid: int | None = None,
    *,
    now_ns: int,
    max_age_ns: int,
) -> Record | None:
    """Newest record for ``metric`` (and ``pid``), or None if absent or older than max_age_ns."""
    best: Record | None = None
    for r in records:
        if r.metric == metric and r.pid == pid and (best is None or r.ts_ns >= best.ts_ns):
            best = r
    if best is None or now_ns - best.ts_ns > max_age_ns:
        return None
    return best


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    name: str
    sm_util: float | None
    vram_mib: float | None
    is_mps: bool | None


def process_rows(
    records: list[Record], names: dict[int, str], now_ns: int, max_age_ns: int
) -> list[ProcessRow]:
    pids = {
        r.pid
        for r in records
        if r.pid is not None
        and r.metric.startswith("proc.")
        and now_ns - r.ts_ns <= max_age_ns
    }

    def val(metric: str, pid: int) -> float | None:
        r = latest(records, metric, pid, now_ns=now_ns, max_age_ns=max_age_ns)
        return None if r is None else r.value

    rows = []
    for pid in pids:
        mps = val("proc.is_mps", pid)
        rows.append(
            ProcessRow(
                pid,
                names.get(pid, "?"),
                val("proc.sm_util", pid),
                val("proc.mem_used", pid),
                None if mps is None else bool(mps),
            )
        )
    rows.sort(key=lambda r: (r.sm_util is None, -(r.sm_util or 0.0), r.pid))
    return rows


def sparkline(values: list[float | None], width: int) -> str:
    """Last ``width`` samples as block characters (0-100 scale); GAP for None."""
    vals = values[-width:] if width > 0 else []
    if not vals:
        return GAP * max(width, 0)
    out = []
    for v in vals:
        if v is None:
            out.append(GAP)
        else:
            idx = round(min(max(v, 0.0), 100.0) / 100.0 * (len(_BLOCKS) - 1))
            out.append(_BLOCKS[idx])
    return "".join(out)


def fmt_vram(mib: float | None) -> str:
    if mib is None:
        return NA
    if mib >= 1024:
        return f"{mib / 1024:.1f}G"
    return f"{mib:.0f}M"


def fmt_temp(c: float | None) -> str:
    return NA if c is None else f"{c:.0f} C"


def fmt_power(w: float | None) -> str:
    return NA if w is None else f"{w:.0f} W"
