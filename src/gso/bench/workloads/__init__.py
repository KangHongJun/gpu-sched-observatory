"""합성 워크로드 공용 도우미 (GPU/torch 불필요)."""

import os
from collections.abc import MutableMapping


def select_gpu(index: int, environ: MutableMapping[str, str] = os.environ) -> None:
    """nvidia-smi와 같은 PCI bus ID 순서로 GPU `index` 하나만 보이게 한다.

    torch를 import하기 전에 호출해야 한다.
    """
    environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environ["CUDA_VISIBLE_DEVICES"] = str(index)


import importlib.util

# python -m gso.bench.workloads.<name>로 실행되는 모듈. trt_infer는 tensorrt 패키지가 있을 때만
# 넣는다(import하지 않고 find_spec으로 확인한다).
_CANDIDATES = ("small_kernels", "membound", "fp16_matmul", "trt_infer", "decode")
WORKLOADS = tuple(
    n for n in _CANDIDATES if n != "trt_infer" or importlib.util.find_spec("tensorrt") is not None
)
