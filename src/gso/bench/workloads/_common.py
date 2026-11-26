"""합성 워크로드 공용 실행 틀 (torch를 import하지 않는다).

CLI 계약: python -m gso.bench.workloads.<name> --gpu N --duration S --out FILE.json
- select_gpu(N)은 torch/CUDA 라이브러리를 import하기 전에 부른다.
- 결과 JSON = summarize(...) + workload + pid (+ 워크로드별 추가 키), 임시 파일 + rename으로 쓴다.
- 실행할 수 없는 환경은 종료 코드 1과 stderr 한 줄, JSON 없음.
- SIGTERM/SIGINT는 측정을 멈추고 아무것도 쓰지 않으며 0이 아닌 코드로 끝난다
  (중간에 끊긴 실행은 측정값이 아니다).
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from gso.analysis.stats import summarize
from gso.bench.workloads import select_gpu

WARMUP_ITERS = 10
EXIT_UNAVAILABLE = 1
EXIT_INTERRUPTED = 130

# (지연 ms 목록, 측정 시간 s, JSON에 더할 추가 키)
Measured = tuple[list[float], float, dict]


class WorkloadUnavailable(Exception):
    """이 장비에서 실행할 수 없는 워크로드(라이브러리 없음, 하드웨어 디코더 없음 등)."""


class StopFlag:
    """SIGTERM/SIGINT가 세우는 중단 플래그."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self, *_args: object) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def install(self) -> None:
        signal.signal(signal.SIGTERM, self.set)
        signal.signal(signal.SIGINT, self.set)


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
    latencies_ms: list[float] = []
    start = time.perf_counter()
    while True:
        if stop.is_set():
            return None
        t0 = time.perf_counter()
        step()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if t1 - start >= duration_s:
            return latencies_ms, t1 - start


def run_cli(
    name: str,
    description: str,
    add_args: Callable[[argparse.ArgumentParser], None],
    measure: Callable[[argparse.Namespace, StopFlag], Measured | None],
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--out", required=True)
    add_args(parser)
    args = parser.parse_args(argv)

    select_gpu(args.gpu)  # torch/CUDA import보다 먼저
    stop = StopFlag()
    stop.install()
    try:
        measured = measure(args, stop)
    except WorkloadUnavailable as exc:
        print(f"{name}: unavailable: {exc}".replace("\n", " "), file=sys.stderr)
        return EXIT_UNAVAILABLE
    if measured is None or stop.is_set():
        print(f"{name}: interrupted, no result written", file=sys.stderr)
        return EXIT_INTERRUPTED
    latencies_ms, elapsed, extra = measured
    result = summarize(latencies_ms, elapsed)
    result.update({"workload": name, "pid": os.getpid()})
    result.update(extra)
    write_json_atomic(args.out, result)
    return 0
