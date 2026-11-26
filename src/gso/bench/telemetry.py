"""Metrics sampled around the benchmark: NVML power/temperature and DCGM profiling means.

``NvmlDcgmTelemetry`` samples the GPU once per second on a background thread for the whole run
and answers ``window_means(start_ns, end_ns)`` afterwards, so each repetition's means cover its
measured window only. Anything that cannot be measured (NVML down, DCGM missing or a field
unsupported, no sample inside the window) is ``None``, never 0. DCGM values are percent (the
provider already multiplied the 0-1 fractions by 100).
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence

from gso.buffer import RingBuffer
from gso.providers.dcgm import DcgmProvider
from gso.providers.nvml import NvmlProvider
from gso.record import Quality, Record, Scope

MEAN_KEYS = {
    "sm_clock_mhz_mean": "gpu.sm_clock",
    "power_w_mean": "gpu.power",
    "sm_active_mean": "dcgm.sm_active",
    "sm_occupancy_mean": "dcgm.sm_occupancy",
    "tensor_active_mean": "dcgm.tensor_active",
    "dram_active_mean": "dcgm.dram_active",
}
_CAPACITY = 200_000


def mean_in_window(records: Sequence[Record], metric: str, start_ns: int, end_ns: int) -> float | None:
    vals = [
        r.value for r in records
        if r.metric == metric and r.value is not None and r.quality is Quality.MEASURED
        and start_ns <= r.ts_ns <= end_ns
    ]
    return sum(vals) / len(vals) if vals else None


class NvmlDcgmTelemetry:
    def __init__(
        self,
        gpu_index: int,
        dcgm_fields: Mapping[int, bool] | None = None,
        dcgmi_cmd: Sequence[str] | None = None,
        dcgm_gpu_index: int | None = None,
        *,
        nvml=None,
        interval_s: float = 1.0,
        popen=subprocess.Popen,
        clock_ns=time.time_ns,
    ):
        self.gpu_index = gpu_index
        self._fields = dict(dcgm_fields or {})
        self._dcgmi_cmd = list(dcgmi_cmd) if dcgmi_cmd else None
        self._dcgm_index = dcgm_gpu_index if dcgm_gpu_index is not None else gpu_index
        self._nvml = nvml
        self._interval_s = interval_s
        self._popen = popen
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._buffer = RingBuffer(_CAPACITY)
        self._probe = NvmlProvider(gpu_index, nvml)
        self._sampler = NvmlProvider(gpu_index, nvml)
        self._dcgm: DcgmProvider | None = None
        self._mod = None  # the NVML module used for the extra per-second reads
        self._handle = None
        self._inited = False
        self._own: Callable[[int], bool] | None = None
        self._foreign: list[tuple[int, frozenset[int]]] = []  # (ts_ns, pids): in memory only
        self._info = {"power_limit_w": None, "perf_state": None}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.errors: list[str] = []

    def start(self) -> None:
        try:
            self._probe.open()
            self._sampler.open()
        except Exception as e:  # noqa: BLE001 - degrade: every NVML-derived value becomes None
            self.errors.append(f"NVML unavailable: {e}")
            return
        self._open_extras()
        self._thread = threading.Thread(target=self._loop, name="gso-bench-nvml", daemon=True)
        self._thread.start()
        if any(self._fields.values()):
            self._dcgm = DcgmProvider(
                self._dcgm_index, f"gpu-{self.gpu_index}", self._buffer, self._fields,
                interval_ms=1000, dcgmi_cmd=self._dcgmi_cmd, popen=self._popen,
                clock_ns=self._clock_ns,
            )
            self._dcgm.start()

    def _open_extras(self) -> None:
        """Own handle for the SM clock, power limit, performance state and compute-process list."""
        try:
            mod = self._nvml
            if mod is None:
                import pynvml as mod
            mod.nvmlInit()
            self._inited = True
            self._mod = mod
            self._handle = mod.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"NVML extras unavailable: {e}")
            return
        with contextlib.suppress(Exception):  # unsupported: stays None
            self._info["power_limit_w"] = mod.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000
        with contextlib.suppress(Exception):
            self._info["perf_state"] = f"P{int(mod.nvmlDeviceGetPerformanceState(self._handle))}"

    def static_info(self) -> dict:
        """Power limit (W) and performance state at the start, None when NVML cannot say."""
        return dict(self._info)

    def set_own_filter(self, own: Callable[[int], bool] | None) -> None:
        """``own(pid)`` says whether a compute process on this GPU belongs to the benchmark.
        Once set, every tick records which other processes are on the GPU (kept as counts in
        results; PIDs are never written anywhere)."""
        self._own = own

    def _sample_extras(self, ts: int) -> None:
        mod, h = self._mod, self._handle
        if h is None:
            return
        with contextlib.suppress(Exception):  # unsupported: no samples, mean None
            clk = float(mod.nvmlDeviceGetClockInfo(h, getattr(mod, "NVML_CLOCK_SM", 1)))
            self._buffer.push(Record(ts, "", "gpu.sm_clock", clk, "MHz", "nvml", Scope.DEVICE,
                                     Quality.MEASURED))
        own = self._own
        if own is None:
            return
        try:
            pids = {p.pid for fn in (mod.nvmlDeviceGetComputeRunningProcesses,
                                     mod.nvmlDeviceGetMPSComputeRunningProcesses)
                    for p in fn(h)}
        except Exception as e:  # noqa: BLE001
            if len(self.errors) < 5:
                self.errors.append(f"NVML process listing failed: {e}")
            return
        foreign = frozenset(p for p in pids if not self._is_own(own, p))
        with self._lock:
            self._foreign.append((ts, foreign))

    @staticmethod
    def _is_own(own, pid) -> bool:
        try:
            return bool(own(pid))
        except Exception:  # noqa: BLE001 - cannot classify: treat as foreign (fail closed)
            return False

    def foreign_in_window(self, start_ns: int, end_ns: int) -> int:
        """How many distinct non-own compute processes were seen on the GPU in the window."""
        with self._lock:
            seen = [pids for ts, pids in self._foreign if start_ns <= ts <= end_ns]
        return len(frozenset().union(*seen)) if seen else 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            ts = self._clock_ns()
            try:
                for r in self._sampler.sample(ts):
                    if r.metric == "gpu.power":
                        self._buffer.push(r)
            except Exception as e:  # noqa: BLE001 - keep sampling; the means just have fewer points
                if len(self.errors) < 5:
                    self.errors.append(f"NVML sample failed: {e}")
            self._sample_extras(ts)
            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._dcgm is not None:
            self._dcgm.stop()
            if self._dcgm.error:
                self.errors.append(f"DCGM: {self._dcgm.error}")
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._inited:
            with contextlib.suppress(Exception):
                self._mod.nvmlShutdown()
            self._inited = False
        for p in (self._probe, self._sampler):
            with contextlib.suppress(Exception):
                p.close()

    def probe(self) -> tuple[float | None, float | None]:
        """(temperature C, utilization %) right now, None for what cannot be read."""
        try:
            recs = {r.metric: r.value for r in self._probe.sample(self._clock_ns()) if r.scope.value == "device"}
        except Exception:  # noqa: BLE001
            return None, None
        return recs.get("gpu.temp"), recs.get("gpu.util")

    def window_means(self, start_ns: int, end_ns: int) -> dict[str, float | None]:
        recs = self._buffer.snapshot()
        return {k: mean_in_window(recs, m, start_ns, end_ns) for k, m in MEAN_KEYS.items()}
