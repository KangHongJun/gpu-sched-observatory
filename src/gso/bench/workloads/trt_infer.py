"""TensorRT 추론 워크로드: torchvision ResNet-50(무작위 가중치), FP16 엔진.

CLI 계약: python -m gso.bench.workloads.trt_infer --gpu N --duration S --out FILE.json [--batch B]
추가 JSON 키: backend("tensorrt"), batch, precision, trt_version, build_s(엔진 캐시를 쓰면 0)

TensorRT가 없거나 엔진을 만들 수 없으면 종료 코드 1과 이유 한 줄, JSON 없음.
PyTorch 추론으로 대체하지 않는다. TensorRT 11은 strongly typed 네트워크만 지원하므로
FP16 엔진은 모델을 FP16으로 ONNX에 내보내 만든다. 엔진은 ~/.cache/gso/에
모델·배치·정밀도·TensorRT 버전을 키로 캐시한다(키가 바뀌면 다시 만든다).
"""

import argparse
import re
import time
from pathlib import Path

from gso.bench.workloads._common import (
    Measured,
    StopFlag,
    WorkloadUnavailable,
    run_cli,
    timed_loop,
)

CACHE_DIR = Path("~/.cache/gso").expanduser()
MODEL = "resnet50"
PRECISION = "fp16"


def cache_key(model: str, batch: int, precision: str, trt_version: str) -> str:
    raw = f"{model}_b{batch}_{precision}_trt{trt_version}"
    return re.sub(r"[^A-Za-z0-9._-]", "_", raw)


def _export_onnx(path: Path, batch: int) -> None:
    import warnings

    import torch
    import torchvision

    model = torchvision.models.resnet50(weights=None).eval().half().cuda()
    x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=torch.float16)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # 레거시 exporter 안내
        torch.onnx.export(
            model, x, str(path), input_names=["input"], output_names=["output"],
            opset_version=17, dynamo=False,
        )


def _build_engine(trt, onnx_path: Path) -> bytes:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise WorkloadUnavailable(f"TensorRT ONNX parse failed: {errs[:200]}")
    config = builder.create_builder_config()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise WorkloadUnavailable("TensorRT engine build failed")
    return bytes(blob)


def _load_or_build(trt, batch: int) -> tuple[bytes, float]:
    key = cache_key(MODEL, batch, PRECISION, trt.__version__)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    plan = CACHE_DIR / f"{key}.plan"
    if plan.exists():
        return plan.read_bytes(), 0.0
    t0 = time.perf_counter()
    onnx_path = CACHE_DIR / f"{key}.onnx"
    try:
        _export_onnx(onnx_path, batch)
        blob = _build_engine(trt, onnx_path)
    except WorkloadUnavailable:
        raise
    except Exception as exc:
        raise WorkloadUnavailable(f"engine build failed: {type(exc).__name__}: {exc}"[:240]) from exc
    finally:
        onnx_path.unlink(missing_ok=True)
    tmp = plan.with_name(plan.name + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(plan)
    return blob, time.perf_counter() - t0


def measure(args: argparse.Namespace, stop: StopFlag) -> Measured | None:
    try:
        import tensorrt as trt  # CUDA_VISIBLE_DEVICES 설정 이후에 import
    except ImportError as exc:
        raise WorkloadUnavailable(f"tensorrt not installed ({exc})") from exc
    import torch

    blob, build_s = _load_or_build(trt, args.batch)
    if stop.is_set():
        return None
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise WorkloadUnavailable("TensorRT engine deserialize failed")
    context = engine.create_execution_context()
    x = torch.randn(args.batch, 3, 224, 224, device="cuda", dtype=torch.float16)
    y = torch.empty(args.batch, 1000, device="cuda", dtype=torch.float16)
    context.set_tensor_address("input", x.data_ptr())
    context.set_tensor_address("output", y.data_ptr())
    stream = torch.cuda.Stream()

    def step() -> None:
        with torch.cuda.stream(stream):
            if not context.execute_async_v3(stream.cuda_stream):
                raise WorkloadUnavailable("TensorRT execution failed")
        stream.synchronize()

    timed = timed_loop(step, args.duration, stop)
    if timed is None:
        return None
    extra = {
        "backend": "tensorrt", "batch": args.batch, "precision": PRECISION,
        "trt_version": trt.__version__, "build_s": build_s,
    }
    return (*timed, extra)


def main(argv: list[str] | None = None) -> int:
    return run_cli(
        "trt_infer", "TensorRT ResNet-50 FP16 inference workload",
        lambda p: p.add_argument("--batch", type=int, default=8), measure, argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
