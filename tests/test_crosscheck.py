"""Cross-check gso's loader and metrics against independent SQL over the raw nsys export.

The SQL below is written from docs/verification/nsys-schema.md (sections 2-5, 10), not from the
loader. It opens the fixture read-only and never touches PROCESSES. These show gso reads the same
rows nsys exported; they do not validate the scheduling interpretation. The window-rule comparison
restates the documented rule (weaker evidence); the raw_counts comparison is a low-value
consistency check.
"""

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from gso.analysis.intervals import kernel_durations, restores_per_second
from gso.trace.nsys_sqlite import capture_window, load_ctx_events, load_kernels
from gso.trace.report import build_summary

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
FIXTURES = [FIX / "nsys_ctxsw_min.sqlite", FIX / "nsys_launched_min.sqlite"]
MASKED = 281474976710656  # 1 << 48, schema 2.3
PID = "((globalPid >> 24) & 16777215)"  # schema 2.2
CTX = "GPU_CONTEXT_SWITCH_EVENTS"
KER = "CUPTI_ACTIVITY_KIND_KERNEL"
LOADER_TABLES = (
    CTX,
    "ENUM_GPU_CTX_SWITCH",
    KER,
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "StringIds",
    "DIAGNOSTIC_EVENT",
)

fixtures = pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)


def sql(path, query, params=()):
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return con.execute(query, params).fetchall()
    finally:
        con.close()


def has(path, table):
    return bool(
        sql(path, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
    )


@fixtures
def test_ctx_event_count(path):
    assert len(load_ctx_events(path)) == sql(path, f"SELECT COUNT(*) FROM {CTX}")[0][0]


@fixtures
def test_kernel_count_and_total_duration(path):
    if not has(path, KER):
        pytest.skip("no kernel table in this fixture")
    kernels = load_kernels(path)
    assert len(kernels) == sql(path, f"SELECT COUNT(*) FROM {KER}")[0][0]
    assert (
        sum(kernel_durations(kernels))
        == sql(path, f"SELECT SUM(end - start) FROM {KER}")[0][0]
    )


@fixtures
def test_per_pid_kernel_count_first_start_last_end(path):
    if not has(path, KER):
        pytest.skip("no kernel table in this fixture")
    kernels = load_kernels(path)
    expected = sql(
        path, f"SELECT {PID}, COUNT(*), MIN(start), MAX(end) FROM {KER} GROUP BY 1"
    )
    assert expected
    assert {k.pid for k in kernels} == {r[0] for r in expected}
    for pid, count, first, last in expected:
        mine = [k for k in kernels if k.pid == pid]
        assert (
            len(mine),
            min(k.start_ns for k in mine),
            max(k.end_ns for k in mine),
        ) == (
            count,
            first,
            last,
        )


def _ctx_counts(path):
    """{(gpuId, decoded pid or None for the masked bucket, tag): count} from SQL."""
    out = {}
    for gpu, pid, masked, tag, n in sql(
        path,
        f"SELECT gpuId, {PID}, globalPid = {MASKED}, tag, COUNT(*) FROM {CTX}"
        " WHERE tag IN (7, 8) GROUP BY 1, 2, 3, 4",
    ):
        out[(gpu, None if masked else pid, tag)] = n
    return out


@fixtures
def test_per_gpu_pid_restore_and_save_counts(path):
    expected = _ctx_counts(path)
    assert expected
    got = {}
    for e in load_ctx_events(path):
        if e.tag in (7, 8):
            key = (e.gpu_id, None if e.global_pid == MASKED else e.pid, e.tag)
            got[key] = got.get(key, 0) + 1
    assert got == expected


def _window_w1(path):
    """Schema 4.3: w0 = 0, w1 = max(stopped row ts, max event ts over every gpuId)."""
    stopped = sql(
        path,
        "SELECT MAX(timestamp) FROM DIAGNOSTIC_EVENT WHERE text = 'Profiling has stopped.'",
    )[0][0]
    last = sql(path, f"SELECT MAX(timestamp) FROM {CTX}")[0][0]
    return max(v for v in (stopped, last) if v is not None)


@fixtures
def test_capture_window_matches_independent_rule(path):
    assert capture_window(path) == (0, _window_w1(path))


def _expected_rates(path):
    """SQL tag-8 count with 0 <= timestamp < w1 per (gpuId, pid), divided by the window seconds.

    The half-open end follows the [t0, t1) definition in gso.analysis.intervals.
    """
    w1 = _window_w1(path)
    rows = sql(
        path,
        f"SELECT gpuId, {PID}, COUNT(*) FROM {CTX} WHERE tag = 8 AND globalPid != {MASKED}"
        " AND timestamp >= 0 AND timestamp < ? GROUP BY 1, 2",
        (w1,),
    )
    return {(g, p): n / (w1 / 1e9) for g, p, n in rows}


def _gso_rates(path):
    rps = restores_per_second(load_ctx_events(path), *capture_window(path))
    return {(g, (gp >> 24) & 0xFFFFFF): v for (g, gp), v in rps.items()}


@fixtures
def test_restores_per_second_equals_sql_count_over_window(path):
    expected = _expected_rates(path)
    assert expected
    got = _gso_rates(path)
    assert got.keys() == expected.keys()
    for k, v in expected.items():
        assert got[k] == pytest.approx(v, rel=1e-12)


def _synthetic(path, events, stopped_at=None):
    """Hand-built export: (tag, globalPid, gpuId, timestamp) rows, optional stopped row."""
    con = sqlite3.connect(path)
    con.executescript(
        f"CREATE TABLE {CTX} (tag INTEGER NOT NULL, vmId INTEGER NOT NULL, seqNo INTEGER NOT NULL,"
        " contextId INTEGER NOT NULL, timestamp INTEGER NOT NULL, globalPid INTEGER, gpuId INTEGER);"
        "CREATE TABLE ENUM_GPU_CTX_SWITCH (id INTEGER PRIMARY KEY, name TEXT, label TEXT);"
        "INSERT INTO ENUM_GPU_CTX_SWITCH VALUES (7,'SAVE_END','Save end'),(8,'RESTORE_START','x');"
        "CREATE TABLE DIAGNOSTIC_EVENT (timestamp INTEGER, text TEXT);"
    )
    for i, (tag, gpid, gpu, ts) in enumerate(events):
        con.execute(
            f"INSERT INTO {CTX} VALUES (?, 0, ?, 1, ?, ?, ?)", (tag, i, ts, gpid, gpu)
        )
    if stopped_at is not None:
        con.execute(
            "INSERT INTO DIAGNOSTIC_EVENT VALUES (?, 'Profiling has stopped.')",
            (stopped_at,),
        )
    con.commit()
    con.close()
    return path


GP = 4242 << 24  # a decoded pid for the hand-built rows (schema 2.2)
SEC = 1_000_000_000


def _check_synthetic(path, expected_count, expected_w1):
    assert _window_w1(path) == expected_w1
    assert capture_window(path) == (0, expected_w1)
    exp = _expected_rates(path)
    assert exp == {(1, 4242): expected_count / (expected_w1 / 1e9)}
    got = _gso_rates(path)
    assert got.keys() == exp.keys()
    assert got[(1, 4242)] == pytest.approx(exp[(1, 4242)], rel=1e-12)


def test_window_end_with_stopped_row_counts_event_at_w1_minus_1ns(tmp_path):
    w1 = 3 * SEC
    ev = [(8, GP, 1, t) for t in (SEC // 2, 2 * SEC + SEC // 2, w1 - 1)]
    path = _synthetic(tmp_path / "a.sqlite", ev, stopped_at=w1)
    _check_synthetic(path, 3, w1)


def test_window_end_without_stopped_row_excludes_event_exactly_at_w1(tmp_path):
    w1 = 3 * SEC  # no stopped row: w1 is the last event itself
    ev = [(8, GP, 1, t) for t in (SEC, 2 * SEC + SEC // 2, w1)]
    path = _synthetic(tmp_path / "b.sqlite", ev)
    _check_synthetic(path, 2, w1)


@fixtures
def test_report_raw_counts_equal_sql_counts(path):
    mode = "launched" if has(path, KER) else "global"
    counts = build_summary(path, gpu=1, mode=mode, masked=True)["raw_counts"]
    expected = {
        t: sql(path, f"SELECT COUNT(*) FROM {t}")[0][0]
        for t in LOADER_TABLES
        if has(path, t)
    }
    assert counts == expected


@fixtures
def test_crosscheck_script_exits_zero_and_prints_no_pid(path):
    pids = {str(r[0]) for r in sql(path, f"SELECT DISTINCT {PID} FROM {CTX}") if r[0]}
    gpu = "1"
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify" / "21_crosscheck.py"),
            str(path),
            "--gpu",
            gpu,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "✗" not in res.stdout
    assert "comparisons match" in res.stdout
    for pid in pids:
        assert pid not in res.stdout


def test_crosscheck_script_on_masked_only_gpu_prints_only_masked_labels():
    path = FIX / "nsys_ctxsw_min.sqlite"
    res = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify" / "21_crosscheck.py"),
            str(path),
            "--gpu",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "GPU 0 ctx, masked:" in res.stdout
    assert "process " not in res.stdout
    assert "restores/s" not in res.stdout
