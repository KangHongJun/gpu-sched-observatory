import pytest

from tests.fakes import FakeNvml


def test_process_shapes():
    f = FakeNvml(procs={0: [10, (11, 2048)]}, mps_procs={0: [(12, 4096)]})
    a, b = f.nvmlDeviceGetComputeRunningProcesses(0)
    assert (a.pid, a.usedGpuMemory) == (10, None)
    assert (b.pid, b.usedGpuMemory) == (11, 2048)
    (m,) = f.nvmlDeviceGetMPSComputeRunningProcesses(0)
    assert (m.pid, m.usedGpuMemory) == (12, 4096)
    assert f.nvmlDeviceGetComputeRunningProcesses(1) == []


def test_errors_injection_class_and_instance():
    f = FakeNvml(
        errors={
            "nvmlInit": FakeNvml.NVMLError_NotFound,
            "nvmlShutdown": RuntimeError("x"),
        }
    )
    with pytest.raises(FakeNvml.NVMLError_NotFound):
        f.nvmlInit()
    with pytest.raises(RuntimeError):
        f.nvmlShutdown()


def test_exception_hierarchy():
    f = FakeNvml()
    assert issubclass(f.NVMLError_NotFound, f.NVMLError)
    assert issubclass(f.NVMLError_NotSupported, f.NVMLError)


def test_calls_recorded_including_failures():
    f = FakeNvml(errors={"nvmlDeviceGetCount": FakeNvml.NVMLError})
    f.nvmlInit()
    with pytest.raises(FakeNvml.NVMLError):
        f.nvmlDeviceGetCount()
    assert f.calls == ["nvmlInit", "nvmlDeviceGetCount"]
