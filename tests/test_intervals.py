"""파생 지표 테스트 (src/gso/analysis/intervals.py)."""

from __future__ import annotations

import random
import statistics
from pathlib import Path

import pytest

from gso.analysis.intervals import (
    context_residency,
    interprocess_overlap,
    kernel_durations,
    launch_to_start_delays,
    restores_per_second,
    slice_durations,
    summarize_ns,
    switch_rate,
    switch_rate_by_gpu,
    union_length,
)
from gso.trace import nsys_sqlite as ns
from gso.trace.nsys_sqlite import CtxEvent, KernelInterval

S = 1_000_000_000
FIXTURES = Path(__file__).parent / "fixtures"
CTXSW = FIXTURES / "nsys_ctxsw_min.sqlite"
LAUNCHED = FIXTURES / "nsys_launched_min.sqlite"
MASK = 1 << 48


def gp(pid: int) -> int:
    return (1 << 48) | (pid << 24)


def ev(ts, who, *, gpu=0, tag=8):
    """who: 정수 pid, 또는 None(마스킹 행)."""
    if who is None:
        return CtxEvent(gpu, 0, MASK, None, "X", tag, ts)
    return CtxEvent(gpu, who, gp(who), who, "X", tag, ts)


def ctx(ts, context_id):
    return ev(ts, context_id)


def kernel(api_start, start, end, pid=1):
    return KernelInterval(pid, 0, 1, 7, start, end, api_start, "k", None)


# ------------------------------------------------------------ brief 예제


def test_union_and_residency():
    iv = [(0, 10), (5, 15), (20, 30)]
    assert union_length(iv) == 25
    assert context_residency(iv, 0, 40) == 0.625


def test_switch_rate_counts_only_changes():
    e = [ctx(0, 1), ctx(S // 2, 1), ctx(S, 2), ctx(3 * S // 2, 1)]
    assert switch_rate(e, 0, 2 * S) == 1.0


def test_overlap_only_across_pids():
    assert interprocess_overlap({1: [(0, 10)], 2: [(5, 15)]}, 0, 20) == 0.25
    assert interprocess_overlap({1: [(0, 10), (5, 15)]}, 0, 20) is None
    assert interprocess_overlap({1: [(0, 5)], 2: [(10, 15)]}, 0, 20) == 0.0


def test_empty_and_zero_window_return_none():
    assert switch_rate([], 0, S) is None
    assert context_residency([(0, 10)], 5, 5) is None
    assert interprocess_overlap({}, 0, 0) is None
    assert switch_rate([ctx(0, 1)], 5, 5) is None


def test_launch_to_start_skips_missing_api_start():
    ks = [kernel(100, 150, 200), kernel(None, 10, 20)]
    assert launch_to_start_delays(ks) == [50]


# ------------------------------------------------------------ switch rate


def test_save_end_ignored():
    e = [ev(0, 1), ev(10, 1, tag=7), ev(20, 2, tag=7), ev(S, 2)]
    assert switch_rate(e, 0, 2 * S) == 0.5  # RESTORE_START 1 -> 2, 전환 1회 / 2초


def test_same_context_twice_not_a_switch():
    e = [ev(0, 1), ev(100, 1), ev(200, 1)]
    assert switch_rate(e, 0, S) == 0.0


def test_masked_is_one_opaque_context():
    assert switch_rate([ev(0, None), ev(S // 4, 1), ev(S // 2, None)], 0, S) == 2.0
    assert switch_rate([ev(0, None), ev(S // 4, None)], 0, S) == 0.0


def test_first_restore_per_gpu_is_not_a_switch():
    assert switch_rate([ev(0, 1)], 0, S) == 0.0


def test_two_gpus_counted_separately_and_summed():
    e = [
        ev(0, 1, gpu=0), ev(10, 2, gpu=1), ev(20, 2, gpu=0), ev(30, 1, gpu=1),
        ev(40, 1, gpu=0), ev(50, 1, gpu=1),
    ]  # fmt: skip
    # gpu0: 1,2,1 -> 2 switches; gpu1: 2,1,1 -> 1 switch
    assert switch_rate(e, 0, S) == 3.0
    assert switch_rate_by_gpu(e, 0, S) == {0: 2.0, 1: 1.0}


def test_window_is_half_open():
    e = [ev(10, 1), ev(20, 2), ev(30, 1)]
    assert switch_rate(e, 10, 30) == pytest.approx(1 / (20 / 1e9))  # ts=30 excluded
    assert switch_rate(e, 11, 31) == pytest.approx(1 / (20 / 1e9))  # ts=10 excluded
    assert switch_rate(e, 0, 10) is None  # no RESTORE_START inside


def test_restores_per_second_and_by_gpu():
    e = [ev(0, 1), ev(S // 2, 2), ev(S, 1), ev(3 * S // 2, None), ev(S // 4, 1, gpu=1)]
    r = restores_per_second(e, 0, 2 * S)
    assert r == {(0, gp(1)): 1.0, (0, gp(2)): 0.5, (1, gp(1)): 0.5}
    assert restores_per_second(e, 0, 0) == {}
    assert restores_per_second([], 0, S) == {}
    by = switch_rate_by_gpu(e, 0, 2 * S)
    assert by[0] == 1.5 and by[1] == 0.0  # gpu0: 1,2,1,masked -> 3 switches / 2 s


def test_switch_rate_by_gpu_none_cases():
    assert switch_rate_by_gpu([], 0, S) == {}
    assert switch_rate_by_gpu([ev(0, 1)], 5, 5) == {}


# ------------------------------------------------------------ overlap


def test_overlap_three_pids_nested():
    by = {1: [(0, 20)], 2: [(5, 15)], 3: [(8, 12)]}
    assert interprocess_overlap(by, 0, 20) == pytest.approx(10 / 20)  # [5,15)


def test_overlap_same_pid_ignored_other_pid_partial():
    by = {1: [(0, 10), (5, 15)], 2: [(12, 20)]}
    assert interprocess_overlap(by, 0, 20) == pytest.approx(3 / 20)  # [12,15)


def test_overlap_clipped_to_window():
    by = {1: [(-10, 10)], 2: [(5, 30)]}
    assert interprocess_overlap(by, 0, 20) == pytest.approx(5 / 20)  # [5,10)


def test_overlap_touching_is_not_overlap():
    assert interprocess_overlap({1: [(0, 10)], 2: [(10, 20)]}, 0, 20) == 0.0


def test_overlap_single_pid_or_outside_window_is_none():
    assert interprocess_overlap({1: [(0, 10)]}, 0, 20) is None
    assert interprocess_overlap({1: [(0, 10)], 2: [(30, 40)]}, 0, 20) is None
    assert interprocess_overlap({1: [(0, 10)], 2: []}, 0, 20) is None


def test_overlap_matches_brute_force():
    rng = random.Random(1234)
    for _ in range(200):
        by = {}
        for pid in range(rng.randint(1, 4)):
            ivs = []
            for _ in range(rng.randint(0, 4)):
                a = rng.randint(0, 50)
                ivs.append((a, rng.randint(a, 50)))
            by[pid] = ivs
        t0 = rng.randint(0, 40)
        t1 = rng.randint(t0, 50)
        present = {
            p for p, ivs in by.items() if any(min(e, t1) > max(s, t0) for s, e in ivs)
        }
        got = interprocess_overlap(by, t0, t1)
        if t1 <= t0 or len(present) < 2:
            assert got is None
            continue
        covered = 0
        for u in range(t0, t1):
            pids = {p for p, ivs in by.items() if any(s <= u < e for s, e in ivs)}
            covered += len(pids) >= 2
        assert got == pytest.approx(covered / (t1 - t0))


# ------------------------------------------------------------ union / slices


def test_union_and_slices_unsorted_touching_duplicates():
    iv = [(20, 30), (0, 10), (10, 15), (5, 8), (20, 30), (40, 41)]
    assert union_length(iv) == 15 + 10 + 1
    assert slice_durations(iv) == [15, 10, 1]
    assert union_length([]) == 0
    assert slice_durations([]) == []


def test_residency_clips_and_empty():
    assert context_residency([(0, 100)], 25, 75) == 1.0
    assert context_residency([(0, 10), (90, 100)], 5, 95) == pytest.approx(10 / 90)
    assert context_residency([], 0, 10) == 0.0
    assert context_residency([(0, 10)], 20, 30) == 0.0


def test_kernel_durations_and_delays_order():
    ks = [kernel(None, 0, 7), kernel(10, 30, 40), kernel(0, 5, 6)]
    assert kernel_durations(ks) == [7, 10, 1]
    assert launch_to_start_delays(ks) == [20, 5]


def test_validation():
    with pytest.raises(ValueError):
        context_residency([(0, 1)], 10, 5)
    with pytest.raises(ValueError):
        switch_rate([], 10, 5)
    with pytest.raises(ValueError):
        switch_rate_by_gpu([], 10, 5)
    with pytest.raises(ValueError):
        restores_per_second([], 10, 5)
    with pytest.raises(ValueError):
        interprocess_overlap({}, 10, 5)
    with pytest.raises(ValueError):
        union_length([(5, 4)])
    with pytest.raises(ValueError):
        slice_durations([(5, 4)])
    with pytest.raises(ValueError):
        context_residency([(5, 4)], 0, 10)
    with pytest.raises(ValueError):
        interprocess_overlap({1: [(5, 4)]}, 0, 10)


def test_inputs_not_mutated():
    iv = [(20, 30), (0, 10), (5, 15)]
    by = {2: [(9, 12), (0, 3)], 1: [(5, 15)]}
    e = [ev(5, 1), ev(0, 2)]
    ks = [kernel(1, 2, 3)]
    iv0, by0, e0, ks0 = list(iv), {k: list(v) for k, v in by.items()}, list(e), list(ks)
    union_length(iv)
    slice_durations(iv)
    context_residency(iv, 0, 40)
    interprocess_overlap(by, 0, 20)
    switch_rate(e, 0, 10)
    restores_per_second(e, 0, 10)
    kernel_durations(ks)
    launch_to_start_delays(ks)
    assert (iv, by, e, ks) == (iv0, by0, e0, ks0)


# ------------------------------------------------------------ summarize_ns


def test_summarize_ns():
    vals = [1_000_000 * i for i in range(1, 101)]
    s = summarize_ns(vals)
    assert s == {"n": 100, "p50_ms": 50.0, "p95_ms": 95.0, "p99_ms": 99.0}


def test_summarize_ns_empty():
    assert summarize_ns([]) == {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}


# ------------------------------------------------------------ 실제 fixture


def test_real_global_capture():
    w = ns.capture_window(CTXSW)
    events = ns.load_ctx_events(CTXSW)
    r = restores_per_second(events, *w)
    assert len([k for k in r if k[0] == 1]) == 2
    assert switch_rate_by_gpu(events, *w)[1] > 0
    sums = {}
    for cons in ("restore_save", "hold"):
        ivs = ns.ctx_intervals(events, cons, window=w)
        total = 0.0
        for (gpu, _), v in ivs.items():
            if gpu != 1:
                continue
            res = context_residency(v, *w)
            assert 0.0 <= res <= 1.0
            total += res
        sums[cons] = total
    assert sums["restore_save"] < sums["hold"]
    assert interprocess_overlap({}, *w) is None
    assert ns.load_kernels(CTXSW) == []


def test_real_launched_capture():
    w = ns.capture_window(LAUNCHED)
    ks = ns.load_kernels(LAUNCHED)
    assert all(d > 0 for d in kernel_durations(ks))
    delays = launch_to_start_delays(ks)
    assert len(delays) == 2000
    # nsys-schema.md 5.3: kernel.start - runtime.start 최소 4.1 us, 최대 1,377 us
    med_us = statistics.median(delays) / 1e3
    assert 4.1 <= med_us <= 1377
    by_pid: dict[int, list] = {}
    for k in ks:
        by_pid.setdefault(k.pid, []).append((k.start_ns, k.end_ns))
    assert len(by_pid) == 1
    assert interprocess_overlap(by_pid, *w) is None
