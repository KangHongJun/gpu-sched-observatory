"""NVML provider: 틱마다 GPU 한 장을 샘플링해 Record 목록을 돌려준다.

DCGM/nsys가 없어도 항상 쓸 수 있는 기본 소스다. 지원하지 않는 질의나 샘플링 중 사라진
프로세스 때문에 예외가 나지 않도록, 지표 단위로 UNSUPPORTED/DROPPED로 내려앉는다.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from gso.record import Quality, Record, Scope, unsupported

SOURCE = "nvml"
_MIB = 1048576


def _s(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def proc_identity(pid: int, proc_root: str = "/proc") -> tuple[int, int] | None:
    """(pid, starttime). PID 재사용 구분용. 읽을 수 없으면 None."""
    try:
        with open(f"{proc_root}/{pid}/stat", encoding="utf-8", errors="replace") as f:
            text = f.read()
        # 명령 이름(2번 필드)은 공백·괄호를 포함할 수 있으므로 마지막 ')' 뒤부터 파싱한다.
        rest = text[text.rindex(")") + 1 :].split()
        return pid, int(rest[19])  # 전체 22번째 필드 = ')' 뒤 20번째
    except (OSError, ValueError, IndexError):
        return None


class NvmlProvider:
    def __init__(self, gpu_index: int, nvml=None):
        self.gpu_index = gpu_index
        self._nvml = nvml
        self._handle = None
        self._uuid: str | None = None
        self._last_seen_ts = 0

    # -- 수명주기 -------------------------------------------------------
    def open(self) -> None:
        if self._handle is not None:
            return
        if self._nvml is None:
            import pynvml

            self._nvml = pynvml
        self._nvml.nvmlInit()
        try:
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._uuid = _s(self._nvml.nvmlDeviceGetUUID(self._handle))
        except Exception:
            self._handle = None
            self._nvml.nvmlShutdown()
            raise

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle = None
        self._nvml.nvmlShutdown()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- 샘플링 ---------------------------------------------------------
    def sample(self, now_ns: int) -> list[Record]:
        if self._handle is None:
            raise RuntimeError("NvmlProvider.sample() called before open()")
        recs = self._device_records(now_ns)
        recs.extend(self._process_records(now_ns))
        return recs

    def _rec(self, now_ns, metric, value, unit, scope, pid=None, quality=Quality.MEASURED):
        return Record(
            now_ns, self._uuid, metric, value, unit, SOURCE, scope, quality, pid=pid
        )

    def _unsup(self, now_ns, metric, unit, scope, pid=None):
        return unsupported(now_ns, self._uuid, metric, unit, SOURCE, scope, pid=pid)

    def _dropped(self, now_ns, metric, unit, scope, pid=None):
        return self._rec(now_ns, metric, None, unit, scope, pid, Quality.DROPPED)

    def _device_records(self, now_ns: int) -> list[Record]:
        n, h = self._nvml, self._handle
        queries: list[tuple[str, str, Callable[[], float]]] = [
            ("gpu.util", "%", lambda: n.nvmlDeviceGetUtilizationRates(h).gpu),
            ("gpu.mem_used", "MiB", lambda: n.nvmlDeviceGetMemoryInfo(h).used / _MIB),
            ("gpu.power", "W", lambda: n.nvmlDeviceGetPowerUsage(h) / 1000),
            (
                "gpu.temp",
                "C",
                lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
            ),
            ("gpu.enc_util", "%", lambda: n.nvmlDeviceGetEncoderUtilization(h)[0]),
            ("gpu.dec_util", "%", lambda: n.nvmlDeviceGetDecoderUtilization(h)[0]),
        ]
        out = []
        for metric, unit, query in queries:
            try:
                value = float(query())
            except n.NVMLError_NotSupported:
                out.append(self._unsup(now_ns, metric, unit, Scope.DEVICE))
                continue
            except (n.NVMLError, TypeError, ValueError, ArithmeticError):
                value = math.nan
            if math.isfinite(value):
                out.append(self._rec(now_ns, metric, value, unit, Scope.DEVICE))
            else:
                out.append(self._dropped(now_ns, metric, unit, Scope.DEVICE))
        return out

    def _list_processes(self) -> tuple[dict[int, object], set[int]] | None:
        n, h = self._nvml, self._handle
        try:
            compute = list(n.nvmlDeviceGetComputeRunningProcesses(h))
        except n.NVMLError:
            return None
        try:
            mps = list(n.nvmlDeviceGetMPSComputeRunningProcesses(h))
        except n.NVMLError:
            mps = []
        procs: dict[int, object] = {}
        for p in compute + mps:
            procs.setdefault(p.pid, p)
        return procs, {p.pid for p in mps}

    def _sm_samples(self):
        """({pid: sm_util}, 상태). 상태: 'ok' | 'none'(샘플 없음) | 'unsupported'."""
        n = self._nvml
        try:
            samples = n.nvmlDeviceGetProcessUtilization(self._handle, self._last_seen_ts)
        except n.NVMLError_NotFound:
            return {}, "none"
        except n.NVMLError_NotSupported:
            return {}, "unsupported"
        except n.NVMLError:
            return {}, "none"
        latest: dict[int, tuple[int, float]] = {}
        for s in samples:
            if s.pid not in latest or s.timeStamp >= latest[s.pid][0]:
                latest[s.pid] = (s.timeStamp, s.smUtil)
            self._last_seen_ts = max(self._last_seen_ts, s.timeStamp)
        return {pid: sm for pid, (_, sm) in latest.items()}, "ok"

    def _process_records(self, now_ns: int) -> list[Record]:
        listed = self._list_processes()
        if listed is None:
            return []
        procs, mps_pids = listed
        if not procs:
            return []
        sm, state = self._sm_samples()
        n = self._nvml
        out: list[Record] = []
        for pid, p in procs.items():
            P = Scope.PROCESS
            try:
                mem = p.usedGpuMemory
            except n.NVMLError_NotFound:
                continue  # 샘플링 중 사라진 프로세스: 레코드를 하나도 남기지 않는다
            recs = []
            if mem is None:
                recs.append(self._unsup(now_ns, "proc.mem_used", "MiB", P, pid))
            else:
                recs.append(self._rec(now_ns, "proc.mem_used", mem / _MIB, "MiB", P, pid))
            recs.append(
                self._rec(now_ns, "proc.is_mps", 1.0 if pid in mps_pids else 0.0, "", P, pid)
            )
            if state == "unsupported":
                recs.append(self._unsup(now_ns, "proc.sm_util", "%", P, pid))
            elif pid in sm:
                recs.append(self._rec(now_ns, "proc.sm_util", float(sm[pid]), "%", P, pid))
            else:
                recs.append(self._dropped(now_ns, "proc.sm_util", "%", P, pid))
            out.extend(recs)
        return out
