"""Comparison report for an MPS A/B benchmark run (``results.jsonl`` + ``run.json``).

Rules (project-wide): ``None`` is unknown and renders as ``N/A`` (a measured 0 renders as 0);
profiled and unprofiled rows are never mixed in one table; only rows that are ``ok`` and ``fair``
enter the statistics, and excluded rows are counted per (condition, proc), never dropped silently.
Per proc (A, B) the workloads may differ, so there is no A+B total.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from gso.bench.runner import CONDITIONS

NA = "N/A"
BASELINE = "mps_off"
NO_RESULTS = "N/A — no benchmark results loaded (--bench-dir DIR)"
_ORDER = [c[0] for c in CONDITIONS]
_METRICS = {
    "throughput": "throughput_per_s",
    "p50": "p50_ms",
    "p95": "p95_ms",
    "p99": "p99_ms",
}
_LATENCY_KEYS = ("p50_ms", "p95_ms", "p99_ms")


class ReportError(Exception):
    """The results directory cannot be turned into a report (the CLI exits 2)."""


@dataclass(frozen=True)
class Stat:
    mean: float | None
    stdev: float | None
    cv: float | None  # stdev / mean; None for n < 2 or a mean of 0
    n: int


@dataclass(frozen=True)
class Stats:
    workload: str
    n_used: int
    n_excluded: int
    throughput: Stat
    p50: Stat
    p95: Stat
    p99: Stat
    requested_pct: str | None = (
        None  # client_thread_pct_requested (the limit asked for)
    )
    server_pct: str | None = None  # mps_server_active_thread_pct (read from the server)
    exclusions: tuple[str, ...] = field(default_factory=tuple)


# -- loading ---------------------------------------------------------------------------------
def load_results(directory: str | Path) -> tuple[list[dict], int]:
    """Rows of ``DIR/results.jsonl`` and the number of malformed lines that were skipped."""
    d = Path(directory)
    if not d.is_dir():
        raise ReportError(f"not a directory: {d}")
    path = d / "results.jsonl"
    if not path.is_file():
        raise ReportError(f"{path} not found")
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ReportError(f"cannot read {path}: {e}") from None
    rows: list[dict] = []
    skipped = 0
    for raw in data.splitlines():
        try:
            line = raw.decode("utf-8")
            if not line.strip():
                continue
            obj = json.loads(line)
        except (
            ValueError
        ):  # UnicodeDecodeError and JSONDecodeError are both ValueErrors
            skipped += 1
            continue
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("condition"), str)
            and isinstance(obj.get("proc"), str)
        ):
            rows.append(obj)
        else:
            skipped += 1
    if not rows:
        raise ReportError(
            f"{path} has no valid result rows ({skipped} malformed line(s))"
        )
    return rows, skipped


def load_meta(directory: str | Path) -> dict | None:
    """``DIR/run.json`` or None when it is missing or unreadable."""
    try:
        obj = json.loads((Path(directory) / "run.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


# -- statistics ------------------------------------------------------------------------------
def _num(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        return None
    return float(v)


def _is_profiled(row: dict) -> bool:
    return row.get("profiled") is True


def _usable(row: dict) -> bool:
    return row.get("ok") is True and row.get("fair") is True


def _check_unmixed(rows: Sequence[dict]) -> None:
    if len({_is_profiled(r) for r in rows}) > 1:
        raise ValueError(
            "profiled and unprofiled rows must not be mixed; split them first"
        )


def _stat(values: Sequence[float]) -> Stat:
    n = len(values)
    if n == 0:
        return Stat(None, None, None, 0)
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if n > 1 else None
    cv = stdev / mean if stdev is not None and mean != 0 else None
    return Stat(mean, stdev, cv, n)


def _value(row: dict, key: str) -> float | None:
    if key in _LATENCY_KEYS and row.get("metric_kind") == "throughput_only":
        return None  # decode: throughput only
    return _num(row.get(key))


def _joined(rows: Sequence[dict], key: str) -> str | None:
    vals = sorted({v for r in rows if (v := _num(r.get(key))) is not None})
    return "/".join(f"{v:g}" for v in vals) if vals else None


def _why_excluded(row: dict) -> str:
    return str(
        row.get("fair_reason")
        or row.get("error")
        or ("failed repetition" if row.get("ok") is not True else "unfair")
    )


def aggregate(rows: Iterable[dict]) -> dict[tuple[str, str], Stats]:
    """Statistics per (condition, proc) over the ``ok`` and ``fair`` rows.

    Groups whose rows were all excluded are still present (n_used 0, unknown statistics).
    """
    rows = list(rows)
    _check_unmixed(rows)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["condition"], r["proc"]), []).append(r)
    out: dict[tuple[str, str], Stats] = {}
    for key, grp in groups.items():
        used = [r for r in grp if _usable(r)]
        excluded = [r for r in grp if not _usable(r)]
        stats = {
            name: _stat([v for r in used if (v := _value(r, col)) is not None])
            for name, col in _METRICS.items()
        }
        workloads = sorted({str(r["workload"]) for r in grp if r.get("workload")})
        out[key] = Stats(
            workload="/".join(workloads) or NA,
            n_used=len(used),
            n_excluded=len(excluded),
            requested_pct=_joined(grp, "client_thread_pct_requested"),
            server_pct=_joined(grp, "mps_server_active_thread_pct"),
            exclusions=tuple(_why_excluded(r) for r in excluded),
            **stats,
        )
    return out


def _change(value: float | None, base: float | None) -> float | None:
    if value is None or base is None or base == 0:
        return None
    return (value - base) / base * 100


def _sort_key(key: tuple[str, str]) -> tuple:
    cond, proc = key
    return (proc, _ORDER.index(cond) if cond in _ORDER else len(_ORDER), cond)


def compare(rows: Iterable[dict], baseline: str = BASELINE) -> list[dict]:
    """One dict per (proc, condition); change % is ``(value - baseline) / baseline * 100``."""
    agg = aggregate(rows)
    out = []
    for key in sorted(agg, key=_sort_key):
        cond, proc = key
        st = agg[key]
        base = agg.get((baseline, proc))
        b_thr = base.throughput.mean if base and base.n_used else None
        b_p99 = base.p99.mean if base and base.n_used else None
        out.append(
            {
                "condition": cond,
                "proc": proc,
                "workload": st.workload,
                "n": st.n_used,
                "n_excluded": st.n_excluded,
                "requested_pct": st.requested_pct,
                "server_pct": st.server_pct,
                "throughput_n": st.throughput.n,
                "p99_n": st.p99.n,
                "throughput_mean": st.throughput.mean,
                "throughput_stdev": st.throughput.stdev,
                "throughput_cv": st.throughput.cv,
                "p99_mean": st.p99.mean,
                "p99_stdev": st.p99.stdev,
                "p99_cv": st.p99.cv,
                "throughput_change_pct": _change(st.throughput.mean, b_thr),
                "p99_change_pct": _change(st.p99.mean, b_p99),
            }
        )
    return out


# -- rendering -------------------------------------------------------------------------------
def _f(v, spec: str) -> str:
    return NA if v is None else format(v, spec)


def _pct_cell(v: float | None) -> str:
    return NA if v is None else f"{v * 100:.2f}"


def _headers(baseline: str) -> list[str]:
    return [
        "Condition",
        "n used",
        "n excluded",
        "Thread % requested",
        "Thread % server",
        "Throughput/s mean",
        "Throughput CV %",
        f"Throughput change vs {baseline} %",
        "p99 ms mean",
        "p99 CV %",
        f"p99 change vs {baseline} %",
    ]


def _short(c: dict, key: str) -> str:
    """Marker when a mean covers fewer repetitions than 'n used' shows."""
    return f" (n={c[key]})" if 0 < c[key] < c["n"] else ""


def _cells(c: dict) -> list[str]:
    return [
        c["condition"],
        str(c["n"]),
        str(c["n_excluded"]),
        c["requested_pct"] or NA,
        c["server_pct"] or NA,
        _f(c["throughput_mean"], ".1f") + _short(c, "throughput_n"),
        _pct_cell(c["throughput_cv"]),
        _f(c["throughput_change_pct"], "+.1f"),
        _f(c["p99_mean"], ".3f") + _short(c, "p99_n"),
        _pct_cell(c["p99_cv"]),
        _f(c["p99_change_pct"], "+.1f"),
    ]


def _table(headers: list[str], body: list[list[str]], md: bool) -> list[str]:
    if md:
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|",
        ]
        return lines + ["| " + " | ".join(r) + " |" for r in body]
    widths = [max(len(x) for x in col) for col in zip(headers, *body, strict=True)]
    return [
        "  ".join(x.ljust(w) for x, w in zip(r, widths, strict=True)).rstrip()
        for r in [headers, *body]
    ]


def _section(rows: Sequence[dict], baseline: str, md: bool) -> list[str]:
    lines: list[str] = []
    comps = compare(rows, baseline)
    for proc in sorted({c["proc"] for c in comps}):
        mine = [c for c in comps if c["proc"] == proc]
        workload = "/".join(sorted({c["workload"] for c in mine}))
        title = f"Proc {proc}: {workload}"
        lines += ["", f"### {title}" if md else title, ""]
        lines += _table(_headers(baseline), [_cells(c) for c in mine], md)
    return lines


def _temps(rows: Sequence[dict]) -> str:
    by_cond: dict[str, list[float]] = {}
    for r in rows:
        t = _num(r.get("gpu_temp_start_c"))
        if t is not None:
            by_cond.setdefault(r["condition"], []).append(t)
    if not by_cond:
        return f"gpu_temp_start_c: {NA}"
    allv = [t for v in by_cond.values() for t in v]
    per = ", ".join(
        f"{c} {min(v):g}-{max(v):g}"
        for c, v in sorted(by_cond.items(), key=lambda kv: _sort_key((kv[0], "")))
    )
    return f"gpu_temp_start_c ranged {min(allv):g}-{max(allv):g} C across conditions ({per} C)"


def _notes(rows: Sequence[dict], meta: dict | None, skipped: int) -> list[str]:
    notes = [
        "Conditions ran in a fixed order, so GPU thermal drift is confounded with condition "
        "(it was not randomised); " + _temps(rows) + ".",
    ]
    if meta is None:
        notes.append(
            "run.json was not found or is unreadable: run completeness, DCGM availability, "
            f"contamination events and telemetry errors are {NA}."
        )
    elif meta.get("complete") is False:
        msg = f" ({meta['message']})" if meta.get("message") else ""
        notes.append(
            f"Run INCOMPLETE: status {meta.get('status') or NA}{msg}. Missing repetitions "
            "are not in the table."
        )
    elif meta.get("complete") is None:
        notes.append(f"Run completeness is {NA} (run.json has no `complete` field).")
    timeouts = {
        (r["condition"], r.get("rep"))
        for r in rows
        if r.get("cooldown_timeout") is True
    }
    if timeouts:
        notes.append(
            f"The cooldown did not reach the target temperature before {len(timeouts)} "
            "repetition(s) (timed out)."
        )
    if meta is not None and meta.get("dcgm_available") is False:
        reason = str(meta.get("dcgm_reason") or NA).rstrip(". ")
        notes.append(f"DCGM unavailable: {reason}. DCGM metrics are {NA}.")
    events = (meta or {}).get("contamination_events") or []
    if events:
        foreign = sum(
            1
            for r in rows
            if r.get("fair") is False
            and "foreign" in str(r.get("fair_reason") or "").lower()
        )
        tail = (
            f" {foreign} row(s) are marked unfair for it and excluded."
            if foreign
            else ""
        )
        notes.append(
            f"{len(events)} contamination event(s): a foreign GPU process was seen.{tail}"
        )
    errs = (meta or {}).get("telemetry_errors") or []
    if errs:
        notes.append(f"{len(errs)} telemetry error(s) were recorded during the run.")
    server = {
        v
        for r in rows
        if (v := _num(r.get("mps_server_active_thread_pct"))) is not None
    }
    requested = {
        v for r in rows if (v := _num(r.get("client_thread_pct_requested"))) is not None
    }
    if len(server) == 1 and len(requested) > 1:
        notes.append(
            f"MPS server active thread percentage (server) stayed at {next(iter(server)):g} "
            "regardless of the per-client thread percentage requested ("
            + ", ".join(f"{v:g}" for v in sorted(requested))
            + ")."
        )
    if skipped:
        notes.append(f"{skipped} malformed line(s) in results.jsonl were skipped.")
    why = Counter(_why_excluded(r) for r in rows if not _usable(r))
    if why:
        notes.append(
            "Excluded from the statistics (not ok or not fair): "
            + "; ".join(f"{n} row(s): {w}" for w, n in why.most_common())
            + "."
        )
    if any(_is_profiled(r) for r in rows):
        notes.append(
            "Profiled rows come from on-demand capture runs; capture overhead makes them "
            "not comparable with unprofiled rows."
        )
    return notes


def _document(
    rows: Sequence[dict], meta: dict | None, skipped: int, baseline: str, md: bool
) -> str:
    rows = list(rows)

    def h1(t: str) -> str:
        return f"# {t}" if md else t

    def h2(t: str) -> str:
        return f"## {t}" if md else t

    lines = [h1("MPS A/B comparison")]
    if meta:
        parts = [
            f"run {meta.get('run_id') or NA}",
            f"GPU {meta.get('gpu')} {meta.get('gpu_name') or NA}",
            f"duration {_f(meta.get('duration_s'), 'g')} s",
            f"reps {_f(meta.get('reps'), 'd')}",
        ]
        lines += ["", " | ".join(parts)]
    if meta is not None and meta.get("complete") is False:
        warn = f"WARNING: run INCOMPLETE (status: {meta.get('status') or NA}); results below are partial."
        lines += ["", f"**{warn}**" if md else warn]
    for profiled, title in ((False, "Unprofiled runs"), (True, "Profiled runs")):
        part = [r for r in rows if _is_profiled(r) is profiled]
        if not part:
            continue
        lines += ["", h2(title)]
        if profiled:
            lines += [
                "",
                "(on-demand capture rows; not comparable with the unprofiled table)",
            ]
        lines += _section(part, baseline, md)
    lines += ["", h2("Notes"), ""]
    lines += [f"- {n}" for n in _notes(rows, meta, skipped)]
    return "\n".join(lines) + "\n"


def to_markdown(
    rows: Iterable[dict],
    meta: dict | None = None,
    skipped_lines: int = 0,
    baseline: str = BASELINE,
) -> str:
    return _document(list(rows), meta, skipped_lines, baseline, md=True)


def to_text(
    rows: Iterable[dict],
    meta: dict | None = None,
    skipped_lines: int = 0,
    baseline: str = BASELINE,
) -> str:
    """The same report with aligned plain-text tables (TUI F5)."""
    return _document(list(rows), meta, skipped_lines, baseline, md=False)


def view_text(directory: str | Path | None) -> str:
    """Text for the F5 view: the report, or an ``N/A`` line saying why there is none."""
    if directory is None:
        return NO_RESULTS
    try:
        rows, skipped = load_results(directory)
    except ReportError as e:
        return f"N/A — {e}"
    return to_text(rows, load_meta(directory), skipped)
