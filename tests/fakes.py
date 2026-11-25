"""테스트 더블: pynvml 모듈 대역(FakeNvml)과 subprocess.run 대역(fake_run)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace


class NVMLError(Exception):
    pass


class NVMLError_NotSupported(NVMLError):
    pass


class NVMLError_NotFound(NVMLError):
    pass


class NVMLError_Uninitialized(NVMLError):
    pass


def _default_gpu(index: int = 0) -> dict:
    return {
        "name": "NVIDIA RTX A5000",
        "uuid": f"GPU-test-{index:04d}",
        "mig": 0,
    }


class FakeNvml:
    """pynvml 모듈 대역.

    gpus:     [{"name", "uuid", "mig"}, ...] (생략된 키는 기본값)
    procs:    {gpu_index: [pid, ...]}  compute 프로세스
    mps_procs:{gpu_index: [pid, ...]}  MPS compute 프로세스
    errors:   {"함수이름": 예외 인스턴스 또는 클래스}  해당 함수가 raise 한다
    driver:   드라이버 버전 문자열
    as_bytes: True면 문자열을 bytes로 돌려준다(구버전 pynvml 모사)
    calls:    호출된 함수 이름 목록
    """

    NVMLError = NVMLError
    NVMLError_NotSupported = NVMLError_NotSupported
    NVMLError_NotFound = NVMLError_NotFound
    NVMLError_Uninitialized = NVMLError_Uninitialized

    def __init__(
        self,
        gpus: list[dict] | None = None,
        procs: dict[int, list[int]] | None = None,
        mps_procs: dict[int, list[int]] | None = None,
        errors: dict | None = None,
        driver: str = "535.00.00",
        as_bytes: bool = False,
    ):
        if gpus is None:
            gpus = [{}]
        self.gpus = [{**_default_gpu(i), **g} for i, g in enumerate(gpus)]
        self.procs = procs or {}
        self.mps_procs = mps_procs or {}
        self.errors = errors or {}
        self.driver = driver
        self.as_bytes = as_bytes
        self.calls: list[str] = []

    # -- 내부 도우미 ----------------------------------------------------
    def _enter(self, fn: str) -> None:
        self.calls.append(fn)
        err = self.errors.get(fn)
        if err is not None:
            raise err() if isinstance(err, type) else err

    def _s(self, text: str):
        return text.encode() if self.as_bytes else text

    def _gpu(self, handle: int) -> dict:
        return self.gpus[handle]

    # -- pynvml 함수 ----------------------------------------------------
    def nvmlInit(self):
        self._enter("nvmlInit")

    def nvmlShutdown(self):
        self._enter("nvmlShutdown")

    def nvmlSystemGetDriverVersion(self):
        self._enter("nvmlSystemGetDriverVersion")
        return self._s(self.driver)

    def nvmlDeviceGetCount(self):
        self._enter("nvmlDeviceGetCount")
        return len(self.gpus)

    def nvmlDeviceGetHandleByIndex(self, index):
        self._enter("nvmlDeviceGetHandleByIndex")
        return index

    def nvmlDeviceGetName(self, handle):
        self._enter("nvmlDeviceGetName")
        return self._s(self._gpu(handle)["name"])

    def nvmlDeviceGetUUID(self, handle):
        self._enter("nvmlDeviceGetUUID")
        return self._s(self._gpu(handle)["uuid"])

    def nvmlDeviceGetMigMode(self, handle):
        self._enter("nvmlDeviceGetMigMode")
        mode = self._gpu(handle)["mig"]
        return (mode, mode)

    def nvmlDeviceGetMPSComputeRunningProcesses(self, handle):
        self._enter("nvmlDeviceGetMPSComputeRunningProcesses")
        return [SimpleNamespace(pid=p) for p in self.mps_procs.get(handle, [])]

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        self._enter("nvmlDeviceGetComputeRunningProcesses")
        return [SimpleNamespace(pid=p) for p in self.procs.get(handle, [])]


def fake_run(returncode=0, stdout="", stderr="", raises=None):
    """subprocess.run 호환 호출 가능 객체. 호출된 argv를 .argvs 에 기록한다."""

    def run(argv, *args, **kwargs):
        run.argvs.append(list(argv))
        run.kwargs.append(kwargs)
        if raises is not None:
            raise raises() if isinstance(raises, type) else raises
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    run.argvs = []
    run.kwargs = []
    return run
