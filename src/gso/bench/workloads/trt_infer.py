"""TensorRT 추론 워크로드: torchvision ResNet-50(무작위 가중치), FP16 엔진.

CLI 계약: python -m gso.bench.workloads.trt_infer --gpu N --duration S --out FILE.json [--batch B]
추가 JSON 키: backend("tensorrt"), batch, precision, trt_version,
build_s(엔진 준비에 실제로 쓴 시간: 빌드 또는 다른 프로세스의 빌드 대기 포함, 캐시만 읽었으면 0),
build_waited_s(그중 다른 프로세스의 빌드를 기다린 시간)

TensorRT가 없거나 엔진을 만들 수 없으면 종료 코드 1과 이유 한 줄, JSON 없음.
PyTorch 추론으로 대체하지 않는다. TensorRT 11은 strongly typed 네트워크만 지원하므로
FP16 엔진은 모델을 FP16으로 ONNX에 내보내 만든다. 엔진은 ~/.cache/gso/에
모델·배치·정밀도·TensorRT 버전을 키로 캐시한다(키가 바뀌면 다시 만든다).
여러 프로세스가 동시에 시작해도 flock(<key>.lock)으로 한 프로세스만 만들고 나머지는 기다린다.
ONNX/임시 파일 이름에는 PID를 넣는다. 읽을 수 없는 엔진 파일은 낡은 것으로 보고 한 번 다시 만든다.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

from gso.bench.workloads._common import (
    Measured,
    StopFlag,
    WorkloadUnavailable,
    file_lock,
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
            model,
            x,
            str(path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
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


def _build_plan(trt, plan: Path, key: str, batch: int, stop: StopFlag) -> bytes:
    onnx_path = CACHE_DIR / f"{key}.{os.getpid()}.onnx"
    tmp = plan.with_name(f"{plan.name}.{os.getpid()}.tmp")
    try:
        try:
            _export_onnx(onnx_path, batch)
            stop.raise_if_set()
            blob = _build_engine(trt, onnx_path)
        except WorkloadUnavailable:
            raise
        except Exception as exc:  # 어떤 실패든 이유 한 줄로 보고
            raise WorkloadUnavailable(
                f"engine build failed: {type(exc).__name__}: {exc}"[:240]
            ) from exc
        stop.raise_if_set()
        tmp.write_bytes(blob)
        tmp.replace(plan)
        return blob
    finally:
        onnx_path.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)


def _load_engine(trt, runtime, batch: int, stop: StopFlag):
    """캐시된 엔진을 역직렬화한다. 없거나 읽을 수 없으면 (잠금 안에서) 한 번 만든다.

    반환: (engine, build_s, build_waited_s).
    """
    key = cache_key(MODEL, batch, PRECISION, trt.__version__)
    plan = CACHE_DIR / f"{key}.plan"

    def note() -> None:
        print("trt_infer: waiting for another process's engine build", file=sys.stderr)

    t0 = time.perf_counter()
    built = False
    with file_lock(CACHE_DIR / f"{key}.lock", stop, note) as waited:
        engine = None
        if plan.exists():
            engine = runtime.deserialize_cuda_engine(plan.read_bytes())
            if engine is None:  # 낡은/손상된 엔진: 지우고 다시 만든다
                plan.unlink(missing_ok=True)
        if engine is None:
            blob = _build_plan(trt, plan, key, batch, stop)
            built = True
            engine = runtime.deserialize_cuda_engine(blob)
            if engine is None:
                plan.unlink(missing_ok=True)
                raise WorkloadUnavailable(
                    "TensorRT engine deserialize failed after rebuild"
                )
    build_s = time.perf_counter() - t0 if (built or waited > 0) else 0.0
    return engine, build_s, waited


def measure(args: argparse.Namespace, stop: StopFlag) -> Measured | None:
    try:
        import tensorrt as trt  # CUDA_VISIBLE_DEVICES 설정 이후에 import
    except ImportError as exc:
        raise WorkloadUnavailable(f"tensorrt not installed ({exc})") from exc
    import torch

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine, build_s, waited_s = _load_engine(trt, runtime, args.batch, stop)
    stop.raise_if_set()
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
        "backend": "tensorrt",
        "batch": args.batch,
        "precision": PRECISION,
        "trt_version": trt.__version__,
        "build_s": build_s,
        "build_waited_s": waited_s,
    }
    return (*timed, extra)


def main(argv: list[str] | None = None) -> int:
    return run_cli(
        "trt_infer",
        "TensorRT ResNet-50 FP16 inference workload",
        lambda p: p.add_argument("--batch", type=int, default=8),
        measure,
        argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
