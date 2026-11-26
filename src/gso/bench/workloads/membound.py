"""메모리 대역폭 제한 워크로드: 큰 텐서 두 개의 복사·덧셈 반복.

CLI 계약: python -m gso.bench.workloads.membound --gpu N --duration S --out FILE.json [--size-mib M]
추가 JSON 키: gbps
"""

import argparse
import statistics

from gso.bench.workloads._common import Measured, StopFlag, run_cli, timed_loop


def measure(args: argparse.Namespace, stop: StopFlag) -> Measured | None:
    import torch  # CUDA_VISIBLE_DEVICES 설정 이후에 import

    device = torch.device("cuda:0")
    n = args.size_mib * 1024 * 1024
    src = torch.empty(n, dtype=torch.uint8, device=device).random_(0, 7)
    dst = torch.empty(n, dtype=torch.uint8, device=device)

    def step() -> None:
        dst.copy_(src)  # 읽기 n + 쓰기 n
        dst.add_(src)  # 읽기 2n(dst, src) + 쓰기 n
        torch.cuda.synchronize()

    timed = timed_loop(step, args.duration, stop)
    if timed is None:
        return None
    latencies_ms, elapsed = timed
    # 추정치: 반복당 이동 바이트 = 5n (copy 읽기1+쓰기1, add 읽기2+쓰기1, 각 접근을 한 번씩만 센다;
    # 캐시 효과와 하드웨어 실제 트랜잭션 수는 반영하지 않는다) / 평균 지연.
    mean_s = statistics.fmean(latencies_ms) / 1000.0
    gbps = 5 * n / mean_s / 1e9
    return latencies_ms, elapsed, {"gbps": gbps, "size_mib": args.size_mib}


def main(argv: list[str] | None = None) -> int:
    return run_cli(
        "membound",
        "memory-bound copy+add workload",
        lambda p: p.add_argument("--size-mib", type=int, default=1024),
        measure,
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
