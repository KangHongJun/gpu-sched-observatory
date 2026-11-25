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
    procs:    {gpu_index: [pid | (pid, used_memory_bytes), ...]}  compute 프로세스.
              반환 객체는 .pid, .usedGpuMemory 를 가진다(bare pid는 None, 실제 pynvml이
              값을 알 수 없을 때 돌려주는 것과 같다).
    mps_procs:procs 와 같은 형태, MPS compute 프로세스
    errors:   {"함수이름": 예외 인스턴스 또는 클래스} 해당 함수가 항상 raise 한다.
              {"함수이름": {gpu_index: 예외}} 형태면 그 장치 핸들에서만 raise 한다.
    참고: 가짜 예외 클래스는 메시지 문자열을 받지만 실제 pynvml.NVMLError 는
          정수 코드를 받는다. 대역일 뿐이므로 의도적으로 다르다.
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
    def _enter(self, fn: str, handle=None) -> None:
        self.calls.append(fn)
        err = self.errors.get(fn)
        if isinstance(err, dict):
            err = err.get(handle)
        if err is not None:
            raise err() if isinstance(err, type) else err

    def _s(self, text: str):
        return text.encode() if self.as_bytes else text

    @staticmethod
    def _procs(table: dict, handle: int) -> list:
        out = []
        for item in table.get(handle, []):
            pid, mem = item if isinstance(item, tuple) else (item, None)
            out.append(SimpleNamespace(pid=pid, usedGpuMemory=mem))
        return out

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
        self._enter("nvmlDeviceGetHandleByIndex", index)
        return index

    def nvmlDeviceGetName(self, handle):
        self._enter("nvmlDeviceGetName", handle)
        return self._s(self._gpu(handle)["name"])

    def nvmlDeviceGetUUID(self, handle):
        self._enter("nvmlDeviceGetUUID", handle)
        return self._s(self._gpu(handle)["uuid"])

    def nvmlDeviceGetMigMode(self, handle):
        self._enter("nvmlDeviceGetMigMode", handle)
        mode = self._gpu(handle)["mig"]
        return (mode, mode)

    def nvmlDeviceGetMPSComputeRunningProcesses(self, handle):
        self._enter("nvmlDeviceGetMPSComputeRunningProcesses", handle)
        return self._procs(self.mps_procs, handle)

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        self._enter("nvmlDeviceGetComputeRunningProcesses", handle)
        return self._procs(self.procs, handle)


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
