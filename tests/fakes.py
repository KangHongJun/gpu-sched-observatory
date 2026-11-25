"""테스트 더블: pynvml 모듈 대역(FakeNvml)과 subprocess.run 대역(fake_run)."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace
from typing import ClassVar


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
    util/mem_used/power_mw/temp/enc/dec: 장치 지표 값(스칼라 또는 {gpu_index: 값}).
              util=(gpu%, mem%), mem_used=바이트, power_mw=밀리와트, enc/dec=퍼센트.
    proc_util:{gpu_index: [(pid, sm_util, timestamp), ...]} nvmlDeviceGetProcessUtilization 결과.
              비어 있으면 NotFound. 받은 lastSeenTimeStamp는 .proc_util_args 에 기록.
    driver:   드라이버 버전 문자열
    as_bytes: True면 문자열을 bytes로 돌려준다(구버전 pynvml 모사)
    calls:    호출된 함수 이름 목록
    """

    NVMLError = NVMLError
    NVMLError_NotSupported = NVMLError_NotSupported
    NVMLError_NotFound = NVMLError_NotFound
    NVMLError_Uninitialized = NVMLError_Uninitialized
    NVML_TEMPERATURE_GPU = 0

    def __init__(
        self,
        gpus: list[dict] | None = None,
        procs: dict[int, list[int]] | None = None,
        mps_procs: dict[int, list[int]] | None = None,
        errors: dict | None = None,
        driver: str = "535.00.00",
        as_bytes: bool = False,
        util: tuple[float, float] | dict = (37, 12),
        mem_used: int | dict = 2 * 1024**3,
        mem_total: int = 24 * 1024**3,
        power_mw: float | dict = 123456,
        temp: float | dict = 55,
        enc: float | dict = 4,
        dec: float | dict = 2,
        proc_util: dict | None = None,
    ):
        if gpus is None:
            gpus = [{}]
        self.gpus = [{**_default_gpu(i), **g} for i, g in enumerate(gpus)]
        self.procs = procs or {}
        self.mps_procs = mps_procs or {}
        self.errors = errors or {}
        self.driver = driver
        self.as_bytes = as_bytes
        self.util = util
        self.mem_used = mem_used
        self.mem_total = mem_total
        self.power_mw = power_mw
        self.temp = temp
        self.enc = enc
        self.dec = dec
        self.proc_util = proc_util or {}
        self.calls: list[str] = []
        self.proc_util_args: list[int] = []

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

    @staticmethod
    def _per_gpu(value, handle: int):
        """스칼라면 그대로, {gpu_index: 값}이면 해당 장치 값."""
        return value[handle] if isinstance(value, dict) else value

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

    def nvmlDeviceGetUtilizationRates(self, handle):
        self._enter("nvmlDeviceGetUtilizationRates", handle)
        gpu, memory = self._per_gpu(self.util, handle)
        return SimpleNamespace(gpu=gpu, memory=memory)

    def nvmlDeviceGetMemoryInfo(self, handle):
        self._enter("nvmlDeviceGetMemoryInfo", handle)
        used = self._per_gpu(self.mem_used, handle)
        return SimpleNamespace(total=self.mem_total, used=used, free=self.mem_total - used)

    def nvmlDeviceGetPowerUsage(self, handle):
        self._enter("nvmlDeviceGetPowerUsage", handle)
        return self._per_gpu(self.power_mw, handle)

    def nvmlDeviceGetTemperature(self, handle, sensor):
        self._enter("nvmlDeviceGetTemperature", handle)
        return self._per_gpu(self.temp, handle)

    def nvmlDeviceGetEncoderUtilization(self, handle):
        self._enter("nvmlDeviceGetEncoderUtilization", handle)
        return [self._per_gpu(self.enc, handle), 167000]

    def nvmlDeviceGetDecoderUtilization(self, handle):
        self._enter("nvmlDeviceGetDecoderUtilization", handle)
        return [self._per_gpu(self.dec, handle), 167000]

    def nvmlDeviceGetProcessUtilization(self, handle, last_seen):
        """proc_util: {gpu_index: [(pid, sm, timestamp), ...]}. 비면 실제 NVML처럼 NotFound."""
        self.proc_util_args.append(last_seen)
        self._enter("nvmlDeviceGetProcessUtilization", handle)
        rows = self.proc_util.get(handle, [])
        if not rows:
            raise NVMLError_NotFound("no samples")
        return [
            SimpleNamespace(pid=pid, timeStamp=ts, smUtil=sm, memUtil=0, encUtil=0, decUtil=0)
            for pid, sm, ts in rows
        ]


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


class FakeCollector:
    """Stands in for gso.collector.Collector; instances are kept in ``made``."""

    made: ClassVar[list] = []
    records_on_start: ClassVar[list] = []
    nvml_error = None
    dcgm_error = None

    def __init__(self, buffer, caps, gpu_index, **kw):
        self.buffer = buffer
        self.starts = 0
        self.stops = 0
        FakeCollector.made.append(self)

    def start(self):
        self.starts += 1
        for r in FakeCollector.records_on_start:
            self.buffer.push(r)

    def stop(self):
        self.stops += 1

    @property
    def status(self):
        return SimpleNamespace(nvml_error=FakeCollector.nvml_error,
                               dcgm_error=FakeCollector.dcgm_error)


class FakeTime:
    """Injected clock and sleep: sleeping advances the clock; ``on_sleep`` may push records."""

    def __init__(self, on_sleep=None):
        self.now = 0.0
        self.sleeps = []
        self.on_sleep = on_sleep

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s
        if self.on_sleep:
            self.on_sleep(len(self.sleeps))
