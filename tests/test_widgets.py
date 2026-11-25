from gso.record import Quality, Record, Scope, unsupported
from gso.tui.widgets import (
    ProcessRow,
    bar,
    fmt_power,
    fmt_temp,
    fmt_vram,
    latest,
    process_rows,
    sparkline,
)

NOW = 100_000_000_000
AGE = 3_000_000_000


def rec(metric, value, ts=NOW, pid=None, quality=Quality.MEASURED, scope=Scope.DEVICE):
    if value is None:
        quality = Quality.DROPPED if quality is Quality.MEASURED else quality
    return Record(ts, "GPU-x", metric, value, "%", "t", scope, quality, pid=pid)


def test_bar():
    assert bar(78.0) == "[############----] 78%"
    assert "N/A" in bar(None) and "0%" not in bar(None)


def test_bar_clamps_and_shows_number():
    assert bar(-5) == "[" + "-" * 16 + "] -5%"
    assert bar(250) == "[" + "#" * 16 + "] 250%"


def test_bar_none_same_length():
    assert len(bar(None)) == len(bar(50.0))
    assert len(bar(None, 10)) == len(bar(50.0, 10))


def test_latest_absent_stale_fresh_pid():
    recs = [
        rec("a", 1.0, ts=NOW - 1),
        rec("a", 2.0, ts=NOW - 10),
        rec("a", 7.0, ts=NOW, pid=5),
        rec("b", 3.0, ts=NOW - AGE - 1),
    ]
    assert latest(recs, "zz", None, now_ns=NOW, max_age_ns=AGE) is None
    assert latest(recs, "b", None, now_ns=NOW, max_age_ns=AGE) is None
    assert latest(recs, "a", None, now_ns=NOW, max_age_ns=AGE).value == 1.0
    assert latest(recs, "a", 5, now_ns=NOW, max_age_ns=AGE).value == 7.0
    assert latest(recs, "a", 6, now_ns=NOW, max_age_ns=AGE) is None


def test_latest_newest_by_timestamp_not_order():
    recs = [rec("a", 2.0, ts=NOW - 1), rec("a", 1.0, ts=NOW - 5)]
    assert latest(recs, "a", None, now_ns=NOW, max_age_ns=AGE).value == 2.0


def test_latest_returns_unknown_record_as_is():
    r = unsupported(NOW, "g", "a", "%", "t", Scope.DEVICE)
    assert latest([r], "a", None, now_ns=NOW, max_age_ns=AGE).value is None


def prec(metric, value, pid, ts=NOW):
    return rec(metric, value, ts=ts, pid=pid, scope=Scope.PROCESS)


def test_process_rows_sort_none_and_names():
    recs = [
        prec("proc.sm_util", 8.0, 10),
        prec("proc.mem_used", 2048.0, 10),
        prec("proc.sm_util", 62.0, 20),
        prec("proc.mem_used", 512.0, 20),
        prec("proc.mem_used", 100.0, 5),
        prec("proc.sm_util", None, 5),
        prec("proc.is_mps", 1.0, 20),
        prec("proc.is_mps", 0.0, 10),
    ]
    rows = process_rows(recs, {10: "dec", 20: "det"}, NOW, AGE)
    assert [r.pid for r in rows] == [20, 10, 5]
    assert rows[0] == ProcessRow(20, "det", 62.0, 512.0, True)
    assert rows[1].is_mps is False
    assert rows[2].sm_util is None and rows[2].name == "?" and rows[2].is_mps is None


def test_process_rows_ties_by_pid_and_stale_omitted():
    recs = [
        prec("proc.sm_util", 5.0, 9),
        prec("proc.sm_util", 5.0, 3),
        prec("proc.sm_util", 50.0, 4, ts=NOW - AGE - 1),
        prec("proc.mem_used", 1.0, 4, ts=NOW - AGE - 1),
    ]
    assert [r.pid for r in process_rows(recs, {}, NOW, AGE)] == [3, 9]


def test_process_rows_stale_field_is_none_not_old_value():
    recs = [prec("proc.sm_util", 50.0, 4, ts=NOW - AGE - 1), prec("proc.mem_used", 9.0, 4)]
    (row,) = process_rows(recs, {}, NOW, AGE)
    assert row.sm_util is None and row.vram_mib == 9.0


def test_sparkline():
    assert len(sparkline([1, 2, 3], 10)) == 3
    s = sparkline([0, None, 100], 3)
    assert s[1] == "·" and s[0] == "▁" and s[2] == "█"
    assert sparkline([], 5) == "·" * 5
    assert sparkline(list(range(100)), 8) == sparkline(list(range(92, 100)), 8)
    assert sparkline([None, None], 4) == "··"


def test_sparkline_clamps():
    assert sparkline([-10, 500], 2) == "▁█"


def test_formatters():
    assert fmt_vram(2457.6) == "2.4G"
    assert fmt_vram(512) == "512M"
    assert fmt_vram(None) == "N/A"
    assert fmt_temp(64.0) == "64 C"
    assert fmt_temp(None) == "N/A"
    assert fmt_power(178.4) == "178 W"
    assert fmt_power(None) == "N/A"


import pytest


@pytest.mark.parametrize("v", [float("nan"), float("inf"), float("-inf")])
def test_bar_non_finite_is_na(v):
    assert "N/A" in bar(v)
    assert len(bar(v)) == len(bar(50.0))
    assert len(bar(v, 10)) == len(bar(50.0, 10))
