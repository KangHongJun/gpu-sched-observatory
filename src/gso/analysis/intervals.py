"""Context/kernel 구간에서 계산하는 파생 지표(순수 함수).

알 수 없음과 0: 입력으로 값을 정할 수 없으면(빈 관찰 구간, 필요한 종류의 이벤트가
없음, 겹침을 재려는데 프로세스가 둘 미만) None을 돌려주고, 입력이 측정된 0을
확정할 때만 0 / 0.0을 돌려준다. 구간은 반열린 ``[start, end)`` ns 튜플이다.
관찰 구간 ``t0 > t1``이거나 ``end < start``인 구간은 ValueError다(``t0 == t1``은
허용하며 해당하는 곳에서 None). 함수는 입력을 바꾸지 않는다.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

from gso.analysis.stats import percentile
from gso.trace.nsys_sqlite import CtxEvent, KernelInterval

_RESTORE_START = 8
_OPAQUE = "masked"  # 한 GPU의 모든 마스킹 행은 하나의 불투명한 컨텍스트


def _check_window(t0: int, t1: int) -> None:
    if t0 > t1:
        raise ValueError(f"window start {t0} is after end {t1}")


def _check_intervals(intervals: Iterable[tuple[int, int]]) -> None:
    for s, e in intervals:
        if e < s:
            raise ValueError(f"interval end {e} is before start {s}")


def _merge(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ivs = list(intervals)
    _check_intervals(ivs)
    merged: list[tuple[int, int]] = []
    for s, e in sorted(ivs):
        if merged and s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def union_length(intervals: list[tuple[int, int]]) -> int:
    """구간들의 합집합 길이(ns). 입력은 정렬·겹침 무관, 빈 입력은 0."""
    return sum(e - s for s, e in _merge(intervals))


def context_residency(
    intervals: list[tuple[int, int]], t0: int, t1: int
) -> float | None:
    """주어진 구간들의 합집합 길이를 관찰 구간 ``[t0, t1)``로 나눈 값.

    구간은 창으로 자른 뒤 합친다. 이 수의 의미는 구간을 만든 구성(``restore_save`` 또는
    ``hold``)에 달려 있다: 두 구성은 둘 다 추정값으로 따로 보여 주며(RESULTS 결정 2)
    이 함수는 어느 쪽이 맞는지 정하지 않고, 결과를 "활성"이나 "실행" 시간이라고
    부르지 않는다. ``t1 <= t0``이면 None. 구간이 없으면(창은 유효) 0.0: 그 컨텍스트에
    구간이 없다는 것이 관측된 것이다.
    """
    _check_window(t0, t1)
    _check_intervals(intervals)
    if t1 <= t0:
        return None
    clipped = [
        (max(s, t0), min(e, t1)) for s, e in intervals if min(e, t1) > max(s, t0)
    ]
    return union_length(clipped) / (t1 - t0)


def slice_durations(intervals: list[tuple[int, int]]) -> list[int]:
    """맞닿거나 겹친 구간을 합친 뒤 각 구간의 길이(ns), 시간순."""
    return [e - s for s, e in _merge(intervals)]


def _restores(events: list[CtxEvent], t0: int, t1: int) -> list[CtxEvent]:
    return [ev for ev in events if ev.tag == _RESTORE_START and t0 <= ev.ts_ns < t1]


def _context_of(ev: CtxEvent):
    return _OPAQUE if ev.pid is None else ev.global_pid


def _switches_by_gpu(events: list[CtxEvent], t0: int, t1: int) -> dict[int, int]:
    per_gpu: dict[int, list[CtxEvent]] = {}
    for ev in _restores(events, t0, t1):
        per_gpu.setdefault(ev.gpu_id, []).append(ev)
    out: dict[int, int] = {}
    for gpu, evs in per_gpu.items():
        evs.sort(key=lambda e: e.ts_ns)  # 안정 정렬: 같은 ts는 입력 순서
        out[gpu] = sum(
            1 for a, b in itertools.pairwise(evs) if _context_of(a) != _context_of(b)
        )
    return out


def switch_rate(events: list[CtxEvent], t0: int, t1: int) -> float | None:
    """GPU 전체의 컨텍스트 전환 수 / 초.

    ``t0 <= ts_ns < t1``의 RESTORE_START(tag 8)만 센다(SAVE_END 등 다른 태그는 무시).
    GPU별로 시간순에서 직전 RESTORE_START와 컨텍스트(``global_pid``)가 다르면 전환 1회다.
    한 GPU의 마스킹 행(pid None)은 모두 하나의 불투명한 컨텍스트로 본다. 각 GPU의 첫
    RESTORE_START는 전환이 아니다. 모든 GPU의 전환 합을 ``(t1 - t0)`` 초로 나눈다.
    ``t1 <= t0``이거나 창 안에 RESTORE_START가 없으면 None.
    """
    _check_window(t0, t1)
    if t1 <= t0:
        return None
    sw = _switches_by_gpu(events, t0, t1)
    if not sw:
        return None
    return sum(sw.values()) / ((t1 - t0) / 1e9)


def switch_rate_by_gpu(
    events: list[CtxEvent], t0: int, t1: int
) -> dict[int, float | None]:
    """``switch_rate``와 같은 정의를 GPU별로 계산한다.

    창 안에 RESTORE_START가 있는 GPU만 키가 된다. ``t1 <= t0``이면 빈 dict(알 수 없음).
    """
    _check_window(t0, t1)
    if t1 <= t0:
        return {}
    secs = (t1 - t0) / 1e9
    return {gpu: n / secs for gpu, n in _switches_by_gpu(events, t0, t1).items()}


def restores_per_second(
    events: list[CtxEvent], t0: int, t1: int
) -> dict[tuple[int, int], float]:
    """``(gpu_id, global_pid)``별 RESTORE_START 수 / 초(프로세스 단위 값).

    ``t0 <= ts_ns < t1``의 RESTORE_START(tag 8)만 세고 마스킹 행은 제외한다. 창이
    비었거나 길이 0이면 빈 dict.
    """
    _check_window(t0, t1)
    if t1 <= t0:
        return {}
    counts: dict[tuple[int, int], int] = {}
    for ev in _restores(events, t0, t1):
        if ev.pid is None or ev.global_pid is None:
            continue
        key = (ev.gpu_id, ev.global_pid)
        counts[key] = counts.get(key, 0) + 1
    secs = (t1 - t0) / 1e9
    return {k: n / secs for k, n in counts.items()}


def interprocess_overlap(
    by_pid: dict[int, list[tuple[int, int]]], t0: int, t1: int
) -> float | None:
    """서로 다른 PID 둘 이상의 구간이 동시에 있는 시간 / ``(t1 - t0)``.

    커널 구간의 겹침이며 실제 SM 단위 동시 실행량과 같지 않다(명세). 같은 PID의 여러
    스트림이 겹치는 것은 세지 않는다. 구간은 창으로 먼저 자른다(sweep line,
    O(n log n)). ``t1 <= t0``이거나 창 안에 구간이 있는 PID가 둘 미만이면 None: 추적한
    프로세스가 하나뿐이면 프로세스 간 겹침은 알 수 없다. PID가 둘 이상 있는데 겹침이
    없으면 0.0.
    """
    _check_window(t0, t1)
    for ivs in by_pid.values():
        _check_intervals(ivs)
    if t1 <= t0:
        return None
    points: list[tuple[int, int, int]] = []  # (time, delta, pid)
    present: set[int] = set()
    for pid, ivs in by_pid.items():
        for s, e in _merge(ivs):  # 같은 PID 안의 겹침을 먼저 합친다
            s, e = max(s, t0), min(e, t1)
            if e > s:
                present.add(pid)
                points.append((s, 1, pid))
                points.append((e, -1, pid))
    if len(present) < 2:
        return None
    points.sort(key=lambda p: (p[0], p[1]))  # 같은 시각에서는 끝(-1)이 먼저
    active = 0
    last = t0
    total = 0
    for t, delta, _ in points:
        if active >= 2:
            total += t - last
        active += delta
        last = t
    return total / (t1 - t0)


def kernel_durations(kernels: list[KernelInterval]) -> list[int]:
    """커널별 ``end_ns - start_ns``(ns), 입력 순서."""
    out = []
    for k in kernels:
        if k.end_ns < k.start_ns:
            raise ValueError(f"kernel end {k.end_ns} is before start {k.start_ns}")
        out.append(k.end_ns - k.start_ns)
    return out


def launch_to_start_delays(kernels: list[KernelInterval]) -> list[int]:
    """CUDA API 호출 시작에서 커널 시작까지의 간격(ns): ``start_ns - api_start_ns``.

    ``api_start_ns``가 없는 커널은 건너뛰고 입력 순서를 유지한다. 동기 실행에서는 기다리는
    큐가 없지만, 비동기 실행이나 경합에서는 앞선 작업 뒤에서 기다린 시간도 함께 들어간다.
    """
    return [k.start_ns - k.api_start_ns for k in kernels if k.api_start_ns is not None]


def summarize_ns(values: list[int]) -> dict:
    """ns 값 목록을 ``{"n", "p50_ms", "p95_ms", "p99_ms"}``로 요약한다(nearest-rank).

    빈 목록은 n=0이고 백분위수는 None.
    """
    ms = [v / 1e6 for v in values]
    return {
        "n": len(values),
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
        "p99_ms": percentile(ms, 99),
    }
