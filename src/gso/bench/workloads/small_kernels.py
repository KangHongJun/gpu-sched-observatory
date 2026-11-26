"""작은 행렬곱 커널을 반복 실행하는 합성 워크로드.

CLI 계약: python -m gso.bench.workloads.<name> --gpu N --duration S --out FILE.json
"""

import argparse

from gso.bench.workloads._common import Measured, StopFlag, run_cli, timed_loop


def measure(args: argparse.Namespace, stop: StopFlag) -> Measured | None:
    import torch  # CUDA_VISIBLE_DEVICES 설정 이후에 import

    device = torch.device("cuda:0")
    a = torch.randn(args.size, args.size, device=device)
    b = torch.randn(args.size, args.size, device=device)

    def step() -> None:
        torch.matmul(a, b)
        torch.cuda.synchronize()

    timed = timed_loop(step, args.duration, stop)
    return None if timed is None else (*timed, {})


def main(argv: list[str] | None = None) -> int:
    return run_cli(
        "small_kernels",
        "small matmul kernels workload",
        lambda p: p.add_argument("--size", type=int, default=256),
        measure,
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
