"""nsys SQLite 로더 테스트. 기준 문서: docs/verification/nsys-schema.md.

PROCESSES 테이블은 어디에서도 조회하지 않는다(fixture에도 없다).
"""

from __future__ import annotations

import itertools
import os
import shutil
import sqlite3
import statistics
from pathlib import Path

import pytest

from gso.trace import nsys_sqlite as ns
from gso.trace.nsys_sqlite import CtxEvent, KernelInterval

FIXTURES = Path(__file__).parent / "fixtures"
CTXSW = FIXTURES / "nsys_ctxsw_min.sqlite"
LAUNCHED = FIXTURES / "nsys_launched_min.sqlite"

MASK = 1 << 48


def gp(pid: int) -> int:
    return (1 << 48) | (pid << 24)


def sql_rows(path: Path, query: str, params: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def ev(ts: int, tag: int, gpu: int, gpid: int, ctx: int = 1) -> CtxEvent:
    pid = None if gpid == MASK else (gpid >> 24) & 0xFFFFFF
    return CtxEvent(
        gpu_id=gpu,
        context_id=ctx,
        global_pid=gpid,
        pid=pid,
        kind="x",
        tag=tag,
        ts_ns=ts,
    )


def make_db(path: Path, rows: list[tuple], diag: list[tuple] | None = None) -> Path:
    """rows: (tag, seqNo, contextId, timestamp, globalPid, gpuId)."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE GPU_CONTEXT_SWITCH_EVENTS (tag INTEGER NOT NULL, vmId INTEGER NOT NULL,"
        " seqNo INTEGER NOT NULL, contextId INTEGER NOT NULL, timestamp INTEGER NOT NULL,"
        " globalPid INTEGER, gpuId INTEGER)"
    )
    conn.execute(
        "CREATE TABLE ENUM_GPU_CTX_SWITCH (id INTEGER PRIMARY KEY, name TEXT, label TEXT)"
    )
    conn.executemany(
        "INSERT INTO ENUM_GPU_CTX_SWITCH VALUES (?,?,?)",
        [(7, "SAVE_END", "Save end"), (8, "RESTORE_START", "Restore start")],
    )
    conn.execute(
        "CREATE TABLE DIAGNOSTIC_EVENT (timestamp INTEGER NOT NULL, timestampType INTEGER,"
        " source INTEGER, severity INTEGER, text TEXT, globalPid INTEGER)"
    )
    conn.executemany(
        "INSERT INTO GPU_CONTEXT_SWITCH_EVENTS VALUES (?,0,?,?,?,?,?)", rows
    )
    conn.executemany("INSERT INTO DIAGNOSTIC_EVENT VALUES (?,1,3,1,?,NULL)", diag or [])
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------- loaders: ctx


def test_ctx_event_count_matches_raw_counts():
    events = ns.load_ctx_events(CTXSW)
    assert len(events) == ns.raw_counts(CTXSW)["GPU_CONTEXT_SWITCH_EVENTS"] == 11490
    assert len(ns.load_ctx_events(LAUNCHED)) == 7


def test_ctx_events_sorted_by_time():
    for path in (CTXSW, LAUNCHED):
        ts = [e.ts_ns for e in ns.load_ctx_events(path)]
        assert ts == sorted(ts)


def test_gpu_filter():
    only1 = ns.load_ctx_events(CTXSW, gpu_id=1)
    assert len(only1) == 11440
    assert {e.gpu_id for e in only1} == {1}
    assert ns.load_ctx_events(CTXSW, gpu_id=5) == []


def test_decoded_pids_global_fixture():
    events = ns.load_ctx_events(CTXSW)
    expected = {
        r[0]
        for r in sql_rows(
            CTXSW,
            "SELECT DISTINCT (globalPid >> 24) & 16777215 FROM GPU_CONTEXT_SWITCH_EVENTS"
            " WHERE gpuId = 1 AND globalPid != ?",
            (MASK,),
        )
    }
    gpu1 = {e.pid for e in events if e.gpu_id == 1}
    assert None not in gpu1
    assert len(gpu1) == 2
    assert gpu1 == expected
    masked_sql = sql_rows(
        CTXSW,
        "SELECT count(*) FROM GPU_CONTEXT_SWITCH_EVENTS WHERE globalPid = ?",
        (MASK,),
    )[0][0]
    masked = [e for e in events if e.pid is None]
    assert len(masked) == masked_sql == 50  # §9
    assert all(e.global_pid == MASK and e.gpu_id == 0 for e in masked)
    assert all(e.pid is not None for e in events if e.global_pid != MASK)


def test_kind_names_from_enum_table():
    events = ns.load_ctx_events(CTXSW)
    by_tag = {e.tag: e.kind for e in events}
    assert by_tag[8] == "RESTORE_START"
    assert by_tag[7] == "SAVE_END"
    # 이름은 export의 enum 표에서 읽는다: 표를 바꾸면 따라 바뀐다.
    first = events[0]
    assert (first.tag, first.ts_ns, first.gpu_id) == (8, 259841, 1)  # §9.1 첫 행
    assert first.context_id == 64787


def test_kind_follows_enum_table(tmp_path):
    db = make_db(tmp_path / "t.sqlite", [(8, 0, 5, 10, gp(7), 1)])
    conn = sqlite3.connect(db)
    conn.execute("UPDATE ENUM_GPU_CTX_SWITCH SET name = 'RENAMED' WHERE id = 8")
    conn.commit()
    conn.close()
    assert ns.load_ctx_events(db)[0].kind == "RENAMED"


# -------------------------------------------------------------- loaders: kernel


def test_kernels_valid_and_sorted():
    kernels = ns.load_kernels(LAUNCHED)
    assert len(kernels) == 2000 == ns.raw_counts(LAUNCHED)["CUPTI_ACTIVITY_KIND_KERNEL"]
    assert all(k.start_ns < k.end_ns for k in kernels)
    keys = [(k.start_ns, k.end_ns) for k in kernels]
    assert keys == sorted(keys)
    assert all(isinstance(k, KernelInterval) for k in kernels)


def test_kernel_pid_single_and_decoded_independently():
    kernels = ns.load_kernels(LAUNCHED)
    expected = {
        r[0]
        for r in sql_rows(
            LAUNCHED,
            "SELECT DISTINCT (globalPid >> 24) & 16777215 FROM CUPTI_ACTIVITY_KIND_KERNEL",
        )
    }
    assert len(expected) == 1
    assert {k.pid for k in kernels} == expected
    assert {k.device_id for k in kernels} == {0}
    assert {k.context_id for k in kernels} == {1}
    assert {k.stream_id for k in kernels} == {7}


def test_kernel_names_resolved():
    names = {k.name for k in ns.load_kernels(LAUNCHED)}
    assert "distribution_elementwise_grid_stride_kernel" in names
    assert any(n.startswith("ampere_sgemm_32x32_sliced1x4_n") for n in names)


def test_api_start_matches_independent_join():
    kernels = ns.load_kernels(LAUNCHED)
    joined = sql_rows(
        LAUNCHED,
        "SELECT k.start, r.start FROM CUPTI_ACTIVITY_KIND_KERNEL k"
        " LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId",
    )
    sql_frac = sum(1 for _, r in joined if r is not None) / len(joined)
    got_frac = sum(1 for k in kernels if k.api_start_ns is not None) / len(kernels)
    assert got_frac == sql_frac == 1.0  # §5.3: 전부 매칭
    with_api = [k for k in kernels if k.api_start_ns is not None]
    assert all(k.api_start_ns <= k.start_ns for k in with_api)
    median = statistics.median(k.start_ns - k.api_start_ns for k in with_api)
    assert 4_000 <= median <= 6_000  # §5.3: p50 4.9 us


def test_api_start_none_without_runtime_table(tmp_path):
    db = tmp_path / "k.sqlite"
    shutil.copy(LAUNCHED, db)
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE CUPTI_ACTIVITY_KIND_RUNTIME")
    conn.commit()
    conn.close()
    kernels = ns.load_kernels(db)
    assert len(kernels) == 2000
    assert all(k.api_start_ns is None for k in kernels)


def test_api_start_none_without_match(tmp_path):
    db = tmp_path / "k.sqlite"
    shutil.copy(LAUNCHED, db)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId = 193")
    conn.commit()
    conn.close()
    kernels = ns.load_kernels(db)
    assert sum(1 for k in kernels if k.api_start_ns is None) == 1


# ------------------------------------------------------------ absent / invalid


def test_empty_sqlite_returns_empty(tmp_path):
    db = tmp_path / "empty.sqlite"
    sqlite3.connect(db).close()
    assert ns.load_ctx_events(db) == []
    assert ns.load_kernels(db) == []
    assert ns.raw_counts(db) == {}
    assert ns.dropped_hints(db) == []
    assert ns.capture_window(db) == (0, 0)


def test_tables_without_needed_ones(tmp_path):
    db = tmp_path / "other.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE unrelated (a INTEGER)")
    conn.commit()
    conn.close()
    assert ns.load_ctx_events(db) == []
    assert ns.load_kernels(db) == []


def test_global_fixture_has_no_kernels_and_launched_has_few_events():
    assert ns.load_kernels(CTXSW) == []
    assert len(ns.load_ctx_events(LAUNCHED)) == 7


def test_nonexistent_path_raises_and_creates_nothing(tmp_path):
    missing = tmp_path / "nope.sqlite"
    for fn in (
        ns.load_ctx_events,
        ns.load_kernels,
        ns.raw_counts,
        ns.dropped_hints,
        ns.capture_window,
    ):
        with pytest.raises(ValueError):
            fn(missing)
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_non_sqlite_file_raises(tmp_path):
    bad = tmp_path / "bad.sqlite"
    bad.write_bytes(b"this is not an sqlite database, just text\n" * 50)
    for fn in (
        ns.load_ctx_events,
        ns.load_kernels,
        ns.raw_counts,
        ns.dropped_hints,
        ns.capture_window,
    ):
        with pytest.raises(ValueError):
            fn(bad)


def test_directory_raises(tmp_path):
    with pytest.raises(ValueError):
        ns.load_ctx_events(tmp_path)


def test_loaders_never_modify_file(tmp_path):
    for src in (CTXSW, LAUNCHED):
        db = tmp_path / src.name
        shutil.copy(src, db)
        before_bytes = db.read_bytes()
        before_mtime = os.stat(db).st_mtime_ns
        before_dir = sorted(p.name for p in tmp_path.iterdir())
        events = ns.load_ctx_events(db)
        ns.load_ctx_events(db, gpu_id=1)
        ns.load_kernels(db)
        ns.raw_counts(db)
        ns.dropped_hints(db)
        ns.capture_window(db)
        ns.ctx_intervals(events, "restore_save")
        assert db.read_bytes() == before_bytes
        assert os.stat(db).st_mtime_ns == before_mtime
        assert sorted(p.name for p in tmp_path.iterdir()) == before_dir


def test_opened_read_only(tmp_path, monkeypatch):
    seen: list[str] = []
    real = sqlite3.connect

    def spy(database, *a, **kw):
        seen.append(str(database))
        return real(database, *a, **kw)

    monkeypatch.setattr(ns.sqlite3, "connect", spy)
    ns.load_ctx_events(CTXSW)
    assert seen and all("mode=ro" in s for s in seen)


# ---------------------------------------------------------------- raw_counts


def test_raw_counts_only_existing_tables():
    c1 = ns.raw_counts(CTXSW)
    assert c1["GPU_CONTEXT_SWITCH_EVENTS"] == 11490
    assert "CUPTI_ACTIVITY_KIND_KERNEL" not in c1
    assert "PROCESSES" not in c1
    c2 = ns.raw_counts(LAUNCHED)
    assert c2["CUPTI_ACTIVITY_KIND_KERNEL"] == 2000
    assert c2["CUPTI_ACTIVITY_KIND_RUNTIME"] == 2000
    assert c2["GPU_CONTEXT_SWITCH_EVENTS"] == 7
    assert c2["StringIds"] == 6
    assert "PROCESSES" not in c2


# -------------------------------------------------------------- capture window


def test_capture_window_fixtures():
    for path in (CTXSW, LAUNCHED):
        stopped = sql_rows(
            path,
            "SELECT max(timestamp) FROM DIAGNOSTIC_EVENT WHERE text = ?",
            ("Profiling has stopped.",),
        )[0][0]
        max_ev = sql_rows(path, "SELECT max(timestamp) FROM GPU_CONTEXT_SWITCH_EVENTS")[
            0
        ][0]
        w0, w1 = ns.capture_window(path)
        assert w0 == 0
        assert w1 == max(stopped, max_ev)
    assert ns.capture_window(CTXSW) == (0, 10_000_366_207)
    assert ns.capture_window(LAUNCHED) == (0, 15_001_009_225)


def test_capture_window_without_diagnostic_row(tmp_path):
    db = make_db(
        tmp_path / "w.sqlite",
        [(8, 0, 1, 100, gp(7), 1), (7, 1, 1, 250, gp(7), 1), (8, 0, 0, 400, MASK, 0)],
        diag=[(0, "Profiling has started.")],
    )
    assert ns.capture_window(db) == (0, 400)


def test_capture_window_event_after_stop_wins(tmp_path):
    db = make_db(
        tmp_path / "w.sqlite",
        [(8, 0, 1, 150, gp(7), 1)],
        diag=[(0, "Profiling has started."), (100, "Profiling has stopped.")],
    )
    assert ns.capture_window(db) == (0, 150)


def test_capture_window_stop_after_events(tmp_path):
    db = make_db(
        tmp_path / "w.sqlite",
        [(8, 0, 1, 50, gp(7), 1)],
        diag=[(100, "Profiling has stopped.")],
    )
    assert ns.capture_window(db) == (0, 100)


# --------------------------------------------------------------- dropped hints


def test_dropped_hints_launched():
    hints = ns.dropped_hints(LAUNCHED)
    assert len(hints) == 2
    assert any("Some CUPTI events might be missing." in h for h in hints)
    assert any(
        h.startswith("Number of incomplete CUPTI events dropped: 1.") for h in hints
    )
    # 시간 범위 밖이라 버려진 수는 버퍼 유실이 아니다(§6).
    assert not any("outside the report's time range" in h for h in hints)
    texts = {r[0] for r in sql_rows(LAUNCHED, "SELECT text FROM DIAGNOSTIC_EVENT")}
    assert set(hints) <= texts  # 원문 그대로


def test_dropped_hints_global_is_empty():
    assert ns.dropped_hints(CTXSW) == []


# ------------------------------------------------- intervals: schema numbers


def _total(intervals: dict, gpu: int) -> int:
    return sum(len(v) for (g, _), v in intervals.items() if g == gpu)


def _check_sorted_disjoint(intervals: dict, window: tuple[int, int]) -> None:
    w0, w1 = window
    for lst in intervals.values():
        assert lst == sorted(lst)
        for a, b in lst:
            assert w0 <= a < b <= w1
        for (_, b1), (a2, _) in itertools.pairwise(lst):
            assert b1 <= a2


@pytest.mark.parametrize("construction", ["restore_save", "hold"])
def test_fixture_intervals_sorted_non_overlapping(construction):
    events = ns.load_ctx_events(CTXSW)
    window = ns.capture_window(CTXSW)
    iv = ns.ctx_intervals(events, construction, window)
    assert iv
    _check_sorted_disjoint(iv, window)
    # 서로 다른 컨텍스트 구간도 겹치지 않는다(§4 주의 1).
    gpu1 = sorted(p for (g, _), v in iv.items() if g == 1 for p in v)
    for (_, b1), (a2, _) in itertools.pairwise(gpu1):
        assert b1 <= a2
    # 마스킹 행은 키가 되지 않는다.
    assert all(g == 1 for g, _ in iv)


@pytest.mark.parametrize("construction", ["restore_save", "hold"])
def test_launched_fixture_intervals(construction):
    events = ns.load_ctx_events(LAUNCHED)
    window = ns.capture_window(LAUNCHED)
    iv = ns.ctx_intervals(events, construction, window)
    _check_sorted_disjoint(iv, window)
    assert list(iv) and all(g == 1 for g, _ in iv)
    total = sum(b - a for v in iv.values() for a, b in v)
    if construction == "restore_save":
        # §4.4: 닫힌 8->7 쌍 23.1 ms + 마지막 8에서 w1까지(자름)
        assert _total(iv, 1) == 2
        spans = sorted(b - a for v in iv.values() for a, b in v)
        assert spans[0] == 3605455057 - 3582317880
        assert spans[1] == window[1] - 3606172585
    else:
        assert _total(iv, 1) == 1
        assert total == window[1] - 3582317880


def test_fixture_restore_save_documented_counts():
    events = ns.load_ctx_events(CTXSW, gpu_id=1)
    window = ns.capture_window(CTXSW)
    iv, stats = ns.ctx_intervals_with_stats(events, "restore_save", window)
    assert _total(iv, 1) == 5716  # §9
    assert stats.unmatched_saves == 6
    assert stats.reopened_restores == 2
    assert stats.trailing_open == 0
    assert stats.leading_clipped == 0
    assert stats.masked_rows == 0
    # 구간 길이 범위 (§4.4: 최소 35.7 us 이상)
    assert min(b - a for v in iv.values() for a, b in v) >= 35_000
    assert len(iv) == 2


def test_fixture_masked_rows_counted_and_ignored():
    events = ns.load_ctx_events(CTXSW)
    iv, stats = ns.ctx_intervals_with_stats(
        events, "restore_save", ns.capture_window(CTXSW)
    )
    assert stats.masked_rows == 50
    assert _total(iv, 1) == 5716
    assert _total(iv, 0) == 0


def test_launched_tag_counts_clipped_trailing():
    events = ns.load_ctx_events(LAUNCHED)
    _, stats = ns.ctx_intervals_with_stats(
        events, "restore_save", ns.capture_window(LAUNCHED)
    )
    assert stats.trailing_open == 1
    assert stats.masked_rows == 4


# ------------------------------------------------ intervals: hand-built rules

A, B, C, E, F = gp(100), gp(200), gp(300), gp(400), gp(500)
W = (0, 1000)

HAND = [
    ev(10, 8, 1, A),
    ev(20, 7, 1, A),
    ev(30, 8, 1, B),
    ev(40, 7, 1, B),
    ev(45, 3, 1, B),  # 7/8 이외 태그: 건너뜀
    ev(50, 8, 1, A),
    ev(55, 8, 1, A),  # 7 없이 다시 열린 8
    ev(60, 7, 1, A),
    ev(70, 7, 1, B),  # 짝 없는 7
    ev(75, 8, 1, MASK),
    ev(76, 7, 1, MASK),
    ev(900, 8, 1, B),  # 뒤쪽 열린 8
    # GPU 2: 앞쪽 7, 그리고 다른 컨텍스트
    ev(5, 7, 2, C),
    ev(20, 8, 2, C),
    ev(30, 7, 2, C),
    ev(50, 8, 2, E),
    # GPU 3: 마스킹 보유자 뒤의 앞쪽 7
    ev(10, 8, 3, MASK),
    ev(20, 7, 3, F),
    ev(30, 8, 3, F),
]
HAND.sort(key=lambda e: e.ts_ns)


def test_restore_save_hand_built():
    iv, stats = ns.ctx_intervals_with_stats(HAND, "restore_save", W)
    assert iv == {
        (1, A): [(10, 20), (55, 60)],
        (1, B): [(30, 40), (900, 1000)],
        (2, C): [(0, 5), (20, 30)],
        (2, E): [(50, 1000)],
        (3, F): [(0, 20), (30, 1000)],
    }
    assert stats.skipped_other_tags == 1
    assert stats.reopened_restores == 1
    assert stats.unmatched_saves == 1
    assert stats.leading_clipped == 2  # C, F
    assert stats.trailing_open == 3  # B, E, F
    assert stats.masked_rows == 3


def test_hold_hand_built():
    iv, stats = ns.ctx_intervals_with_stats(HAND, "hold", W)
    assert iv == {
        (1, A): [(10, 30), (50, 75)],
        (1, B): [(30, 50), (900, 1000)],
        (2, C): [(0, 50)],
        (2, E): [(50, 1000)],
        (3, F): [(30, 1000)],
    }
    assert stats.skipped_other_tags == 1
    assert stats.repeated_restores == 2  # A의 55, C의 20
    assert stats.leading_clipped == 1  # C만(F의 앞쪽 7은 보유자가 있어 무시)
    assert stats.trailing_open == 3
    assert stats.masked_rows == 3


def test_hold_masked_restore_with_no_holder_then_context():
    evs = [ev(10, 8, 1, MASK), ev(20, 8, 1, A), ev(30, 8, 1, MASK)]
    iv = ns.ctx_intervals(evs, "hold", W)
    assert iv == {(1, A): [(20, 30)]}


def test_masked_only_gives_nothing():
    evs = [ev(10, 8, 1, MASK), ev(20, 7, 1, MASK)]
    assert ns.ctx_intervals(evs, "restore_save", W) == {}
    assert ns.ctx_intervals(evs, "hold", W) == {}


def test_intervals_clipped_to_window():
    evs = [ev(10, 8, 1, A), ev(30, 7, 1, A)]
    assert ns.ctx_intervals(evs, "restore_save", (0, 20)) == {(1, A): [(10, 20)]}
    assert ns.ctx_intervals(evs, "hold", (0, 20)) == {(1, A): [(10, 20)]}


def test_default_window_is_max_event_timestamp():
    evs = [ev(10, 8, 1, A), ev(30, 7, 2, B)]
    iv = ns.ctx_intervals(evs, "restore_save")
    assert iv == {(1, A): [(10, 30)], (2, B): [(0, 30)]}


def test_unsorted_input_is_ordered_per_gpu():
    evs = [ev(20, 7, 1, A), ev(10, 8, 1, A)]
    assert ns.ctx_intervals(evs, "restore_save", W) == {(1, A): [(10, 20)]}


def test_construction_required_and_validated():
    with pytest.raises(TypeError):
        ns.ctx_intervals([], window=W)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        ns.ctx_intervals([], "active", W)  # type: ignore[arg-type]


def test_empty_events():
    assert ns.ctx_intervals([], "restore_save") == {}
    assert ns.ctx_intervals([], "hold") == {}
