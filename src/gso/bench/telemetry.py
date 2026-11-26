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
from collections.abc import Mapping, Sequence

from gso.buffer import RingBuffer
from gso.providers.dcgm import DcgmProvider
from gso.providers.nvml import NvmlProvider
from gso.record import Quality, Record

MEAN_KEYS = {
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
        self._buffer = RingBuffer(_CAPACITY)
        self._probe = NvmlProvider(gpu_index, nvml)
        self._sampler = NvmlProvider(gpu_index, nvml)
        self._dcgm: DcgmProvider | None = None
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
        self._thread = threading.Thread(target=self._loop, name="gso-bench-nvml", daemon=True)
        self._thread.start()
        if any(self._fields.values()):
            self._dcgm = DcgmProvider(
                self._dcgm_index, f"gpu-{self.gpu_index}", self._buffer, self._fields,
                interval_ms=1000, dcgmi_cmd=self._dcgmi_cmd, popen=self._popen,
                clock_ns=self._clock_ns,
            )
            self._dcgm.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for r in self._sampler.sample(self._clock_ns()):
                    if r.metric == "gpu.power":
                        self._buffer.push(r)
            except Exception as e:  # noqa: BLE001 - keep sampling; the means just have fewer points
                if len(self.errors) < 5:
                    self.errors.append(f"NVML sample failed: {e}")
            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._dcgm is not None:
            self._dcgm.stop()
            if self._dcgm.error:
                self.errors.append(f"DCGM: {self._dcgm.error}")
        if self._thread is not None:
            self._thread.join(timeout=5)
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
