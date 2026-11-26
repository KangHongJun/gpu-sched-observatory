"""합성 워크로드 공용 도우미 (GPU/torch 불필요)."""

import os
from collections.abc import MutableMapping


def select_gpu(index: int, environ: MutableMapping[str, str] = os.environ) -> None:
    """nvidia-smi와 같은 PCI bus ID 순서로 GPU `index` 하나만 보이게 한다.

    torch를 import하기 전에 호출해야 한다.
    """
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"] = str(index)


# 존재하고 실행되는 워크로드 모듈 (python -m gso.bench.workloads.<name>).
# trt_infer는 TensorRT가 설치·동작할 때만 여기에 둔다.
WORKLOADS = ("small_kernels", "membound", "fp16_matmul", "trt_infer", "decode")
