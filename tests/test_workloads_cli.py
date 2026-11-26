import subprocess
import sys
import time

import pytest

from gso.bench import workloads

TORCH_BASED = ["small_kernels", "membound", "fp16_matmul", "trt_infer"]
ALL = ["small_kernels", "membound", "fp16_matmul", "trt_infer", "decode"]


def _run(name, *args):
    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "-m", f"gso.bench.workloads.{name}", *args],
        check=False, capture_output=True, text=True, timeout=60,
    )
    return p, time.perf_counter() - t0


@pytest.mark.parametrize("name", ALL)
def test_help_lists_contract_flags_fast(name):
    p, dt = _run(name, "--help")
    assert p.returncode == 0, p.stderr
    for flag in ("--gpu", "--duration", "--out"):
        assert flag in p.stdout
    assert dt < 3.0


@pytest.mark.parametrize("name", ALL)
def test_missing_gpu_exits_2(name):
    p, _ = _run(name, "--duration", "1", "--out", "/nonexistent/x.json")
    assert p.returncode == 2


@pytest.mark.parametrize("name", TORCH_BASED)
def test_import_does_not_load_torch(name):
    code = (
        "import sys, importlib;"
        f"importlib.import_module('gso.bench.workloads.{name}');"
        "assert 'torch' not in sys.modules and 'tensorrt' not in sys.modules"
    )
    p = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_workloads_equals_package_modules():
    import importlib.util
    import pkgutil

    import gso.bench.workloads as pkg

    mods = {m.name for m in pkgutil.iter_modules(pkg.__path__) if not m.name.startswith("_")}
    if importlib.util.find_spec("tensorrt") is None:
        mods.discard("trt_infer")
    assert set(workloads.WORKLOADS) == mods
    assert isinstance(workloads.WORKLOADS, tuple)
    assert workloads.WORKLOADS[0] == "small_kernels"
    for name in workloads.WORKLOADS:
        p, _ = _run(name, "--help")
        assert p.returncode == 0, (name, p.stderr)
