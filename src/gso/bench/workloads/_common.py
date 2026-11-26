"""합성 워크로드 공용 실행 틀 (torch를 import하지 않는다).

CLI 계약: python -m gso.bench.workloads.<name> --gpu N --duration S --out FILE.json
- select_gpu(N)은 torch/CUDA 라이브러리를 import하기 전에 부른다.
- 결과 JSON = summarize(...) + workload + pid (+ 워크로드별 추가 키), 임시 파일 + rename으로 쓴다.
- 실행할 수 없는 환경은 종료 코드 1과 stderr 한 줄, JSON 없음.
- --start-at EPOCH_S(선택): 준비와 워밍업이 끝난 뒤 그 벽시계 시각까지 기다렸다가 측정을 시작한다.
  이미 지났으면 "start barrier missed"로 종료 코드 1, JSON 없음. 두 프로세스가 같은 값을 받으면
  같은 시간 구간을 잰다. 실제 측정 구간은 JSON의 window_start_ns/window_end_ns(epoch ns)에 남는다.
- SIGTERM/SIGINT는 측정을 멈추고 아무것도 쓰지 않으며 0이 아닌 코드로 끝난다
  (중간에 끊긴 실행은 측정값이 아니다).
"""

import argparse
import contextlib
import fcntl
import json
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from gso.analysis.stats import summarize
from gso.bench.workloads import select_gpu

WARMUP_ITERS = 10
EXIT_UNAVAILABLE = 1
EXIT_INTERRUPTED = 130


@dataclass
class ThroughputOnly:
    """지연 분포를 주장하지 않는 워크로드의 결과: 반복 수, 측정 시간 s, 추가 키.

    JSON의 latency_ms는 p50/p95/p99 모두 null, metric_kind는 "throughput_only".
    """

    iterations: int
    elapsed_s: float
    extra: dict = field(default_factory=dict)


# (지연 ms 목록, 측정 시간 s, JSON에 더할 추가 키) 또는 ThroughputOnly
Measured = tuple[list[float], float, dict] | ThroughputOnly


class WorkloadUnavailable(Exception):
    """이 장비에서 실행할 수 없는 워크로드(라이브러리 없음, 하드웨어 디코더 없음 등)."""


class BarrierMissed(Exception):
    """--start-at 시각이 측정 시작 전에 이미 지났다(준비가 예산보다 오래 걸렸다)."""


class Interrupted(Exception):
    """SIGTERM/SIGINT로 중단됨(결과를 쓰지 않는다)."""


class StopFlag:
    """SIGTERM/SIGINT가 세우는 중단 플래그."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self, *_args: object) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_set(self) -> None:
        if self.is_set():
            raise Interrupted

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self.set)
        signal.signal(signal.SIGINT, self.set)


class RunContext:
    """한 번의 CLI 실행이 쓰는 시계와 시작 장벽, 기록된 측정 창(epoch s)."""

    def __init__(
        self, start_at: float | None = None, clock=time.time, sleep=time.sleep
    ):
        self.start_at = start_at
        self.clock = clock
        self.sleep = sleep
        self.window: tuple[float, float] | None = None

    def wait_for_start(self, stop: StopFlag) -> bool:
        """start_at까지 기다린다. 중단되면 False. 이미 지났으면 BarrierMissed."""
        if self.start_at is None:
            return True
        now = self.clock()
        if now > self.start_at:
            raise BarrierMissed(f"start barrier missed by {now - self.start_at:.3f} s")
        return self.wait_until(self.start_at, stop)

    def wait_until(self, when: float, stop: StopFlag) -> bool:
        """Sleep (in short slices) until the epoch time ``when``. False if stopped first."""
        while True:
            left = when - self.clock()
            if left <= 0:
                return True
            if stop.is_set():
                return False
            self.sleep(min(left, 0.05))


_ctx = RunContext()


def current() -> RunContext:
    return _ctx


def record_window(start_s: float, end_s: float) -> None:
    """측정 창(epoch 초)을 기록한다. run_cli가 JSON의 window_*_ns로 쓴다."""
    _ctx.window = (start_s, end_s)


def write_json_atomic(out: str | Path, data: dict) -> None:
    """임시 파일에 쓴 뒤 rename한다. 실패하면 임시 파일을 남기지 않는다."""
    out = Path(out)
    text = json.dumps(data, indent=2)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def file_lock(
    path: Path, stop: StopFlag, on_wait: Callable[[], None]
) -> Iterator[float]:
    """`path`에 대한 배타적 flock. 다른 프로세스가 쥐고 있으면 on_wait()를 한 번 부르고 기다린다.

    기다린 시간(s, 기다리지 않았으면 0.0)을 돌려준다. 기다리는 중 중단되면 Interrupted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        waited = 0.0
        t0 = time.perf_counter()
        notified = False
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not notified:
                    on_wait()
                    notified = True
                stop.raise_if_set()
                time.sleep(0.1)
        if notified:
            waited = time.perf_counter() - t0
        yield waited
    finally:
        os.close(fd)  # flock은 fd를 닫으면 풀린다


def timed_loop(
    step: Callable[[], None],
    duration_s: float,
    stop: StopFlag,
    warmup: int = WARMUP_ITERS,
) -> tuple[list[float], float] | None:
    """`step`(GPU 작업 완료까지 포함)을 워밍업 후 duration_s 동안 반복해 지연(ms)을 잰다.

    중단되면 None(부분 결과는 반환하지 않는다).
    """
    for _ in range(warmup):
        if stop.is_set():
            return None
        step()
    if not _ctx.wait_for_start(stop):  # --start-at: 워밍업 뒤, 측정 시작 직전
        return None
    latencies_ms: list[float] = []
    wall_start = _ctx.clock()
    start = time.perf_counter()
    while True:
        if stop.is_set():
            return None
        t0 = time.perf_counter()
        step()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if t1 - start >= duration_s:
            record_window(wall_start, _ctx.clock())
            return latencies_ms, t1 - start


def _start_at(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive epoch time, got {text}")
    return v


def run_cli(
    name: str,
    description: str,
    add_args: Callable[[argparse.ArgumentParser], None],
    measure: Callable[[argparse.Namespace, StopFlag], Measured | None],
    argv: list[str] | None = None,
    *,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    # allow_abbrev=False: the runner's guard on --gpu/--duration/--out/--start-at must not be
    # bypassable by an abbreviation such as --gp
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--start-at",
        type=_start_at,
        default=None,
        metavar="EPOCH_S",
        help="wall-clock time (epoch seconds) at which the measured loop starts, after setup "
        "and warm-up; exit 1 without a result if it has already passed",
    )
    add_args(parser)
    args = parser.parse_args(argv)

    global _ctx
    select_gpu(args.gpu)  # torch/CUDA import보다 먼저
    stop = StopFlag()
    stop.install()
    _ctx = RunContext(args.start_at, clock, sleep)
    try:
        try:
            measured = measure(args, stop)
        except Interrupted:
            measured = None
        except BarrierMissed as exc:
            print(f"{name}: {exc}".replace("\n", " "), file=sys.stderr)
            return EXIT_UNAVAILABLE
        except WorkloadUnavailable as exc:
            print(f"{name}: unavailable: {exc}".replace("\n", " "), file=sys.stderr)
            return EXIT_UNAVAILABLE
        window = _ctx.window
    finally:
        _ctx = RunContext()
    if measured is None or stop.is_set():
        print(f"{name}: interrupted, no result written", file=sys.stderr)
        return EXIT_INTERRUPTED
    if isinstance(measured, ThroughputOnly):
        elapsed = measured.elapsed_s
        result = {
            "iterations": measured.iterations,
            "throughput_per_s": measured.iterations / elapsed if elapsed > 0 else None,
            "latency_ms": {"p50": None, "p95": None, "p99": None},
        }
        extra = measured.extra
        kind = "throughput_only"
    else:
        latencies_ms, elapsed, extra = measured
        result = summarize(latencies_ms, elapsed)
        kind = "latency"
    result.update({"workload": name, "pid": os.getpid(), "metric_kind": kind})
    result["window_start_ns"] = None if window is None else int(window[0] * 1e9)
    result["window_end_ns"] = None if window is None else int(window[1] * 1e9)
    result.update(extra)
    write_json_atomic(args.out, result)
    return 0
