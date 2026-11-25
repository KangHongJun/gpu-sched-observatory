"""작은 행렬곱 커널을 반복 실행하는 합성 워크로드.

CLI 계약: python -m gso.bench.workloads.<name> --gpu N --duration S --out FILE.json
"""

import argparse
import json
import os
import time
from pathlib import Path

from gso.analysis.stats import summarize
from gso.bench.workloads import select_gpu

WARMUP_ITERS = 10


def run(size: int, duration_s: float) -> tuple[list[float], float]:
    import torch  # CUDA_VISIBLE_DEVICES 설정 이후에 import

    device = torch.device("cuda:0")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    for _ in range(WARMUP_ITERS):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    latencies_ms: list[float] = []
    start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        torch.matmul(a, b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        if t1 - start >= duration_s:
            break
    return latencies_ms, t1 - start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="small matmul kernels workload")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args(argv)

    select_gpu(args.gpu)
    latencies_ms, elapsed = run(args.size, args.duration)
    result = summarize(latencies_ms, elapsed)
    result.update({"workload": "small_kernels", "pid": os.getpid()})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
