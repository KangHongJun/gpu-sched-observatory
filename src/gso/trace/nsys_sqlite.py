"""nsys SQLite export 로더: 컨텍스트 전환 이벤트와 커널 구간을 타입 있는 객체로 읽는다.

기준 문서는 docs/verification/nsys-schema.md이다(절 번호는 그 문서). 이 모듈은 문서가
기록한 것만 구현한다. DB는 항상 읽기 전용으로 열고(파일을 만들거나 바꾸지 않는다),
SQL 값은 모두 바인딩 파라미터로 넘긴다(테이블 이름은 이 모듈의 상수만 쓴다).
``PROCESSES`` 테이블은 조회하지 않는다(PID는 ``globalPid``에서 디코딩, §2.2, §8).

구간 이름은 중립적으로 쓴다. ``restore_save``와 ``hold`` 중
어느 쪽이 컨텍스트의 실제 GPU 점유에 해당하는지는 데이터가 정하지 못한다(§4 주의, RESULTS.md 결정 2).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "Construction",
    "CtxEvent",
    "IntervalStats",
    "KernelInterval",
    "KernelLoadStats",
    "capture_window",
    "ctx_intervals",
    "ctx_intervals_with_stats",
    "dropped_hints",
    "load_ctx_events",
    "load_kernels",
    "load_kernels_with_stats",
    "raw_counts",
]

Construction = Literal["restore_save", "hold"]

TAG_SAVE_END = 7  # ENUM_GPU_CTX_SWITCH SAVE_END (§3.2)
TAG_RESTORE_START = 8  # ENUM_GPU_CTX_SWITCH RESTORE_START (§3.2)
MASKED_GLOBAL_PID = 1 << 48  # 다른 사용자 프로세스의 마스킹된 형태 (§2.3)
STOPPED_TEXT = "Profiling has stopped."  # §4.3 캡처 창 w1

# dropped_hints가 보는 DIAGNOSTIC_EVENT 문구(§6에 인용된 문장의 일부). 소문자 비교.
# "events were discarded ... outside the report's time range"는 §6이 버퍼 유실이
# 아니라고 적었으므로 일부러 넣지 않았다.
_LOSS_PATTERNS = ("might be missing", "events dropped")
_ZERO_DROPPED = re.compile(r"events dropped:\s*0(?!\d)")

_READ_TABLES = (
    "GPU_CONTEXT_SWITCH_EVENTS",
    "ENUM_GPU_CTX_SWITCH",
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "StringIds",
    "DIAGNOSTIC_EVENT",
)


@dataclass(frozen=True, slots=True)
class CtxEvent:
    """``GPU_CONTEXT_SWITCH_EVENTS`` 한 행.

    ``context_id``는 이벤트 테이블의 값이며 CUPTI ``contextId``와 다른 번호 공간이다
    (§3.1). ``global_pid``는 원본 값, ``pid``는 ``(globalPid >> 24) & 0xFFFFFF``로
    디코딩한 값이고 마스킹된 형태(``globalPid == 1<<48``, §2.3)나 NULL이면 None이다.
    ``kind``는 export의 ``ENUM_GPU_CTX_SWITCH.name``(예: ``RESTORE_START``)이며
    표에 없으면 ``TAG_<tag>``이다. ``tag``는 원본 정수.
    """

    gpu_id: int
    context_id: int
    global_pid: int | None
    pid: int | None
    kind: str
    tag: int
    ts_ns: int


@dataclass(frozen=True, slots=True)
class KernelInterval:
    """``CUPTI_ACTIVITY_KIND_KERNEL`` 한 행(GPU 실행 시작/종료, §5.1).

    ``device_id``는 프로세스 안의 CUDA 장치 번호이지 nsys ``gpuId``가 아니다(§5.4).
    ``context_id``는 CUPTI의 값이다(§3.1). ``api_start_ns``는 이 커널을 띄운 CUDA
    런타임 API 호출의 시작 시각이다(``correlationId`` 조인, §5.3). 매칭되는 런타임
    행이 없거나 런타임 테이블이 없으면 None. 제출/큐 대기 시각은 export에 없다(§5.1).
    ``name``은 ``demangledName``(템플릿 인자가 풀린 이름, 커널의 식별자)을
    ``StringIds``로 푼 값이다. 그 값이 NULL이거나 ``StringIds``에 없으면 ``mangledName``,
    그것도 없으면 ``shortName``으로 대신하고, 모두 없으면 ``stringid:<demangledName id>``.
    ``short_name``은 ``shortName`` 열을 푼 값으로 표시 전용이다(템플릿 인자가 없고
    잘려 있을 수 있어 같은 이름끼리 묶는 키로 쓰지 않는다).
    """

    pid: int | None
    device_id: int
    context_id: int
    stream_id: int
    start_ns: int
    end_ns: int
    api_start_ns: int | None
    name: str
    short_name: str | None


@dataclass(frozen=True)
class IntervalStats:
    """``ctx_intervals`` 구성 중 센 예외 상황(§4.3). 모든 gpuId 합계."""

    skipped_other_tags: int = 0  # tag가 7/8이 아닌 행(건너뜀)
    masked_rows: int = 0  # 마스킹 행 중 tag 7/8인 것
    unmatched_saves: int = 0  # restore_save: 짝 없는 7(버림)
    reopened_restores: int = 0  # restore_save: 7 없이 다시 온 8(이전 8 버림)
    leading_clipped: int = 0  # 앞쪽 7로 w0에서 시작한 구간
    trailing_open: int = 0  # 끝에서 열려 있어 w1로 자른 구간
    repeated_restores: int = 0  # hold: 현재 보유자의 8(보유 계속)
    zero_length_dropped: int = 0  # 창으로 자른 뒤 길이가 0이 되어 내보내지 않은 구간


@dataclass(frozen=True)
class KernelLoadStats:
    """``load_kernels_with_stats``가 센 값."""

    kernels: int = 0
    # 같은 correlationId와 같은 프로세스의 런타임 행이 둘 이상이라 api_start_ns를
    # None으로 둔 커널 수
    ambiguous_api_matches: int = 0


# --------------------------------------------------------------------- 접근


def pid_of(global_pid: int | None) -> int | None:
    if global_pid is None or global_pid == MASKED_GLOBAL_PID:
        return None
    return (global_pid >> 24) & 0xFFFFFF


@contextmanager
def _open(path: Path | str) -> Iterator[sqlite3.Connection]:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"not an existing file: {p}")
    uri = p.resolve().as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise ValueError(f"cannot open as SQLite database: {p}: {exc}") from exc
    try:
        try:
            conn.execute("SELECT count(*) FROM sqlite_master").fetchone()
        except sqlite3.DatabaseError as exc:
            raise ValueError(f"not an SQLite database: {p}: {exc}") from exc
        yield conn
    finally:
        conn.close()


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


# ------------------------------------------------------------------- 로더


def load_ctx_events(path: Path | str, gpu_id: int | None = None) -> list[CtxEvent]:
    """모든 ``GPU_CONTEXT_SWITCH_EVENTS`` 행을 시간순(``timestamp, gpuId, seqNo``)으로.

    태그를 거르지 않는다(7/8 외 태그도 그대로). 테이블이 없으면 ``[]``. ``gpu_id``를
    주면 그 GPU만. NULL ``gpuId``는 -1로 읽는다(문서에 관측 없음).
    """
    with _open(path) as conn:
        tables = _tables(conn)
        if "GPU_CONTEXT_SWITCH_EVENTS" not in tables:
            return []
        if "ENUM_GPU_CTX_SWITCH" in tables:
            sql = (
                "SELECT e.timestamp, e.gpuId, e.contextId, e.globalPid, e.tag, n.name"
                " FROM GPU_CONTEXT_SWITCH_EVENTS e"
                " LEFT JOIN ENUM_GPU_CTX_SWITCH n ON n.id = e.tag"
            )
        else:
            sql = (
                "SELECT e.timestamp, e.gpuId, e.contextId, e.globalPid, e.tag, NULL"
                " FROM GPU_CONTEXT_SWITCH_EVENTS e"
            )
        params: tuple = ()
        if gpu_id is not None:
            sql += " WHERE e.gpuId = ?"
            params = (gpu_id,)
        sql += " ORDER BY e.timestamp, e.gpuId, e.seqNo"
        rows = conn.execute(sql, params).fetchall()
    return [
        CtxEvent(
            gpu_id=-1 if gpu is None else gpu,
            context_id=ctx,
            global_pid=gpid,
            pid=pid_of(gpid),
            kind=name if name is not None else f"TAG_{tag}",
            tag=tag,
            ts_ns=ts,
        )
        for ts, gpu, ctx, gpid, tag, name in rows
    ]


def load_kernels(path: Path | str) -> list[KernelInterval]:
    """``CUPTI_ACTIVITY_KIND_KERNEL``을 읽는다. 테이블이 없으면 ``[]``.

    ``load_kernels_with_stats(path)[0]``와 같다.
    """
    return load_kernels_with_stats(path)[0]


def load_kernels_with_stats(
    path: Path | str,
) -> tuple[list[KernelInterval], KernelLoadStats]:
    """``load_kernels``와 같고 ``KernelLoadStats``를 함께 돌려준다.

    정렬은 ``start, end, correlationId, rowid`` 순. ``api_start_ns``는 런타임 테이블이
    있을 때, ``correlationId``가 같고 런타임 행의 ``globalTid``로 디코딩한 프로세스가
    커널의 ``globalPid``로 디코딩한 프로세스와 같은 행의 ``start``다(§2.2, §5.3).
    조건을 만족하는 행이 둘 이상이면 하나를 고르지 않고 None으로 두며 세어 둔다.
    """
    with _open(path) as conn:
        tables = _tables(conn)
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            return [], KernelLoadStats()
        has_names = "StringIds" in tables
        has_runtime = "CUPTI_ACTIVITY_KIND_RUNTIME" in tables
        sql = (
            "SELECT k.globalPid, k.deviceId, k.contextId, k.streamId, k.start, k.end,"
            " {api}, k.demangledName, {names}"
            " FROM CUPTI_ACTIVITY_KIND_KERNEL k"
        ).format(
            api="CASE WHEN r.n = 1 THEN r.s END, COALESCE(r.n > 1, 0)"
            if has_runtime
            else "NULL, 0",
            names="sd.value, sm.value, ss.value" if has_names else "NULL, NULL, NULL",
        )
        if has_runtime:
            sql += (
                " LEFT JOIN (SELECT correlationId,"
                " (globalTid >> 24) & 16777215 AS p, MIN(start) AS s, COUNT(*) AS n"
                " FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY correlationId, p) r"
                " ON r.correlationId = k.correlationId"
                " AND r.p = (k.globalPid >> 24) & 16777215"
            )
        if has_names:
            sql += (
                " LEFT JOIN StringIds sd ON sd.id = k.demangledName"
                " LEFT JOIN StringIds sm ON sm.id = k.mangledName"
                " LEFT JOIN StringIds ss ON ss.id = k.shortName"
            )
        sql += " ORDER BY k.start, k.end, k.correlationId, k.rowid"
        rows = conn.execute(sql).fetchall()
    out = []
    ambiguous = 0
    for gpid, dev, ctx, stream, start, end, api, amb, did, dem, man, short in rows:
        ambiguous += amb
        name = dem if dem is not None else man if man is not None else short
        out.append(
            KernelInterval(
                pid=pid_of(gpid),
                device_id=dev,
                context_id=ctx,
                stream_id=stream,
                start_ns=start,
                end_ns=end,
                api_start_ns=api,
                name=name if name is not None else f"stringid:{did}",
                short_name=short,
            )
        )
    return out, KernelLoadStats(kernels=len(out), ambiguous_api_matches=ambiguous)


def capture_window(path: Path | str) -> tuple[int, int]:
    """캡처 창 ``(w0, w1)`` (§4.3). ``w0 = 0``.

    ``w1 = max("Profiling has stopped." 진단 행의 timestamp, 모든 gpuId(마스킹 포함)의
    이벤트 timestamp 최댓값)``. 진단 행이 없으면 이벤트 timestamp 최댓값. 둘 다
    없으면 ``(0, 0)``.
    """
    with _open(path) as conn:
        tables = _tables(conn)
        max_event = None
        if "GPU_CONTEXT_SWITCH_EVENTS" in tables:
            max_event = conn.execute(
                "SELECT max(timestamp) FROM GPU_CONTEXT_SWITCH_EVENTS"
            ).fetchone()[0]
        stopped = None
        if "DIAGNOSTIC_EVENT" in tables:
            stopped = conn.execute(
                "SELECT max(timestamp) FROM DIAGNOSTIC_EVENT WHERE text = ?",
                (STOPPED_TEXT,),
            ).fetchone()[0]
    candidates = [v for v in (stopped, max_event) if v is not None]
    return (0, max(candidates) if candidates else 0)


def raw_counts(path: Path | str) -> dict[str, int]:
    """이 로더가 읽는 테이블 중 존재하는 것의 행 수."""
    with _open(path) as conn:
        tables = _tables(conn)
        return {
            name: conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in _READ_TABLES
            if name in tables
        }


def dropped_hints(path: Path | str) -> list[str]:
    """유실/누락을 알리는 ``DIAGNOSTIC_EVENT.text``를 원문 그대로, timestamp 순으로.

    §6은 인식 규칙을 따로 정하지 않고 문장만 인용한다. 인용된 문장 중 대소문자 무시로
    ``might be missing``(CUPTI 이벤트 누락 가능) 또는 ``events dropped``(불완전
    CUPTI 이벤트 버림)를 포함하는 것을 고른다. "events were discarded ... outside the
    report's time range"는 §6이 버퍼 유실이 아니라고 명시해 제외한다. 개수가 0인
    문장("events dropped: 0.")은 유실 표시가 아니므로 제외한다.
    """
    with _open(path) as conn:
        if "DIAGNOSTIC_EVENT" not in _tables(conn):
            return []
        rows = conn.execute(
            "SELECT text FROM DIAGNOSTIC_EVENT WHERE text IS NOT NULL ORDER BY timestamp"
        ).fetchall()
    return [
        t
        for (t,) in rows
        if any(p in t.lower() for p in _LOSS_PATTERNS)
        and not _ZERO_DROPPED.search(t.lower())
    ]


# ---------------------------------------------------------------- 구간 구성


def ctx_intervals(
    events: list[CtxEvent],
    construction: Construction,
    *,
    window: tuple[int, int],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """컨텍스트 전환 이벤트에서 구간을 만든다(§4.3). 구성은 호출자가 고른다.

    키는 ``(gpu_id, global_pid)``다. §4.3이 이벤트를 ``(gpuId, globalPid)``로 나누므로
    그 ``global_pid``가 context_key이며(이벤트 테이블의 ``contextId``는 쓰지 않는다),
    마스킹 행은 키가 되지 않는다. 값은 정렬되고 서로 겹치지 않는 ``[시작, 끝)`` ns.
    끝점은 창 ``[w0, w1]``로 자른다.

    - ``restore_save``: 같은 컨텍스트의 RESTORE_START→SAVE_END 구간. 마스킹 행은
      무시한다.
    - ``hold``: 컨텍스트가 다른 컨텍스트의 다음 RESTORE_START까지 쥐고 있다고 보는
      구간. 마스킹 행의 RESTORE_START는 현재 보유자의 보유를 끝낸다.

    어느 쪽도 컨텍스트의 실제 GPU 점유라고 단정하지 않는다. ``window``는 필수이고
    ``capture_window(path)``의 결과를 넘긴다(§4.3의 w1은 진단 행과 모든 gpuId의 이벤트를
    보므로, 걸러낸 이벤트 목록에서는 계산할 수 없다). 길이 0이 된 구간은 내보내지 않고
    ``IntervalStats.zero_length_dropped``로 센다.
    """
    return ctx_intervals_with_stats(events, construction, window=window)[0]


def ctx_intervals_with_stats(
    events: list[CtxEvent],
    construction: Construction,
    *,
    window: tuple[int, int],
) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], IntervalStats]:
    """``ctx_intervals``와 같고, 센 예외 상황(``IntervalStats``)을 함께 돌려준다."""
    if construction not in ("restore_save", "hold"):
        raise ValueError(f"unknown construction: {construction!r}")
    if window is None:
        raise ValueError("window is required: pass capture_window(path)")
    w0, w1 = window

    by_gpu: dict[int, list[CtxEvent]] = {}
    for e in events:
        by_gpu.setdefault(e.gpu_id, []).append(e)

    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    counts = dict.fromkeys(IntervalStats.__dataclass_fields__, 0)

    def emit(key: tuple[int, int], a: int, b: int) -> None:
        a, b = min(max(a, w0), w1), min(max(b, w0), w1)
        if a < b:
            out.setdefault(key, []).append((a, b))
        else:
            counts["zero_length_dropped"] += 1

    for gpu in sorted(by_gpu):
        evs = sorted(by_gpu[gpu], key=lambda e: e.ts_ns)  # 안정 정렬: 동률은 입력 순서
        if construction == "restore_save":
            _restore_save(gpu, evs, w0, w1, emit, counts)
        else:
            _hold(gpu, evs, w0, w1, emit, counts)

    for lst in out.values():
        lst.sort()
    return out, IntervalStats(**counts)


def _restore_save(gpu, evs, w0, w1, emit, counts) -> None:
    opened: dict[tuple[int, int], int] = {}
    seen: set[tuple[int, int]] = set()
    for e in evs:
        if e.tag not in (TAG_SAVE_END, TAG_RESTORE_START):
            counts["skipped_other_tags"] += 1
            continue
        if e.pid is None:  # 마스킹(또는 NULL globalPid): 이 구성에서는 무시
            counts["masked_rows"] += 1
            continue
        key = (gpu, e.global_pid)
        if e.tag == TAG_RESTORE_START:
            if key in opened:
                counts["reopened_restores"] += 1
            opened[key] = e.ts_ns
        elif key in opened:
            emit(key, opened.pop(key), e.ts_ns)
        elif key not in seen:  # 이 gpuId에서 이 컨텍스트의 첫 이벤트가 7
            counts["leading_clipped"] += 1
            emit(key, w0, e.ts_ns)
        else:
            counts["unmatched_saves"] += 1
        seen.add(key)
    for key, t in opened.items():
        counts["trailing_open"] += 1
        emit(key, t, w1)


def _hold(gpu, evs, w0, w1, emit, counts) -> None:
    holder: tuple[int, int] | str | None = None  # None / "masked" / key
    start = w0
    for e in evs:
        if e.tag not in (TAG_SAVE_END, TAG_RESTORE_START):
            counts["skipped_other_tags"] += 1
            continue
        masked = e.pid is None
        if masked:
            counts["masked_rows"] += 1
            if e.tag == TAG_RESTORE_START:
                if isinstance(holder, tuple):
                    emit(holder, start, e.ts_ns)
                if holder != "masked":
                    holder, start = "masked", e.ts_ns
            continue  # 마스킹 7은 hold에 영향이 없다
        key = (gpu, e.global_pid)
        if e.tag == TAG_RESTORE_START:
            if holder == key:
                counts["repeated_restores"] += 1
                continue
            if isinstance(holder, tuple):
                emit(holder, start, e.ts_ns)
            holder, start = key, e.ts_ns
        elif holder is None:  # 보유자가 없을 때 비마스킹 컨텍스트의 앞쪽 7
            counts["leading_clipped"] += 1
            holder, start = key, w0
    if isinstance(holder, tuple):
        counts["trailing_open"] += 1
        emit(holder, start, w1)
