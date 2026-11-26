"""FP16 정방 행렬곱 반복 워크로드 (텐서 파이프를 쓴다; TensorRT가 아니다).

CLI 계약: python -m gso.bench.workloads.fp16_matmul --gpu N --duration S --out FILE.json [--size S]
"""

import argparse

from gso.bench.workloads._common import Measured, StopFlag, run_cli, timed_loop


def measure(args: argparse.Namespace, stop: StopFlag) -> Measured | None:
    import torch  # CUDA_VISIBLE_DEVICES 설정 이후에 import

    device = torch.device("cuda:0")
    a = torch.randn(args.size, args.size, device=device, dtype=torch.float16)
    b = torch.randn(args.size, args.size, device=device, dtype=torch.float16)

    def step() -> None:
        torch.matmul(a, b)
        torch.cuda.synchronize()

    timed = timed_loop(step, args.duration, stop)
    return None if timed is None else (*timed, {"size": args.size})


def main(argv: list[str] | None = None) -> int:
    return run_cli(
        "fp16_matmul",
        "FP16 square matmul workload",
        lambda p: p.add_argument("--size", type=int, default=4096),
        measure,
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
