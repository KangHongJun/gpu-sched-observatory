"""Collection orchestration: owns the providers and feeds a RingBuffer from its own thread.

No GPU library and no Textual import at module level, so it also runs headless. The UI reads
only ``buffer.snapshot()`` and the ``status`` snapshot; it never calls a provider.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from gso.buffer import RingBuffer
from gso.capability import Capabilities
from gso.providers import mps as _mps
from gso.providers.dcgm import FIELDS as _DCGM_FIELDS
from gso.providers.mps import MpsStatus
from gso.record import Scope, unsupported

MPS_REFRESH_NS = 5_000_000_000
_JOIN_TIMEOUT_S = 2.0


@dataclass
class CollectorStatus:
    gpu_uuid: str = ""
    gpu_name: str = ""
    nvml_error: str | None = None
    dcgm_error: str | None = None
    mps: MpsStatus | None = None
    last_sample_ns: int | None = None
    process_names: dict[int, str] = field(default_factory=dict)


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


class Collector:
    def __init__(
        self,
        buffer: RingBuffer,
        caps: Capabilities,
        gpu_index: int,
        interval_s: float = 1.0,
        dcgm_interval_ms: int = 1000,
        dcgmi_cmd: Sequence[str] | None = None,
        dcgm_gpu_index: int | None = None,
        nvml_provider=None,
        dcgm_provider=None,
        mps_query=None,
        clock_ns=time.time_ns,
        proc_root: str = "/proc",
    ) -> None:
        self._buffer = buffer
        self._caps = caps
        self._gpu_index = gpu_index
        self._interval_s = interval_s
        self._clock_ns = clock_ns
        self._proc_root = proc_root
        self._mps_query = mps_query if mps_query is not None else _mps.query
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_open = False
        self._dcgm_started = False
        self._mps_checked_ns: int | None = None

        gpu = next((g for g in caps.gpus if g.index == gpu_index), None)
        self._gpu = gpu
        self._st = CollectorStatus(
            gpu_uuid=gpu.uuid if gpu else "", gpu_name=gpu.name if gpu else ""
        )

        self._nvml = nvml_provider
        self._dcgm = None
        self._dcgm_build_error: str | None = None
        if gpu is not None:
            if self._nvml is None:
                from gso.providers.nvml import NvmlProvider

                self._nvml = NvmlProvider(gpu_index)
            if caps.dcgm.available:
                self._dcgm = dcgm_provider
                if self._dcgm is None:
                    try:
                        from gso.providers.dcgm import DcgmProvider

                        self._dcgm = DcgmProvider(
                            dcgm_gpu_index if dcgm_gpu_index is not None else gpu_index,
                            gpu.uuid,
                            buffer,
                            caps.dcgm.fields,
                            interval_ms=dcgm_interval_ms,
                            dcgmi_cmd=dcgmi_cmd,
                        )
                    except Exception as e:  # noqa: BLE001 - bad option must not crash the UI
                        self._dcgm_build_error = _err(e)

    # -- status -----------------------------------------------------------
    @property
    def status(self) -> CollectorStatus:
        """A copy; safe to read from another thread."""
        with self._lock:
            st = CollectorStatus(**{**self._st.__dict__, "process_names": dict(self._st.process_names)})
        st.dcgm_error = self._dcgm_error()
        return st

    def _dcgm_error(self) -> str | None:
        if not self._caps.dcgm.available:
            return self._caps.dcgm.reason or "DCGM unavailable"
        if self._dcgm is None:
            return self._dcgm_build_error or "DCGM provider not created"
        err = getattr(self._dcgm, "error", None)
        if err:
            return err
        if self._dcgm_started and not getattr(self._dcgm, "running", True):
            return "DCGM provider is not running"
        return None

    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._st, k, v)

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._gpu is None:
            self._set(nvml_error=f"GPU {self._gpu_index} not found")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        if self._dcgm is not None and not self._dcgm_started:
            try:
                self._dcgm.start()
                self._dcgm_started = True
            except Exception as e:  # noqa: BLE001
                self._dcgm_build_error = _err(e)
                self._dcgm = None
        self._thread = threading.Thread(target=self._run, name="gso-collector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None:
            t.join(_JOIN_TIMEOUT_S)
        if self._dcgm is not None and self._dcgm_started:
            self._dcgm_started = False
            with contextlib.suppress(Exception):
                self._dcgm.stop()
        if self._nvml_open and (t is None or not t.is_alive()):
            self._nvml_open = False
            with contextlib.suppress(Exception):
                self._nvml.close()

    # -- loop -------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._tick()
            except Exception as e:  # noqa: BLE001 - the loop must survive anything
                self._set(nvml_error=_err(e))
            self._stop_evt.wait(self._interval_s)

    def _tick(self) -> None:
        now = self._clock_ns()
        try:
            if not self._nvml_open:
                self._nvml.open()
                self._nvml_open = True
            recs = self._nvml.sample(now)
        except Exception as e:  # noqa: BLE001
            self._set(nvml_error=_err(e))
        else:
            for r in recs:
                self._buffer.push(r)
            pids = {r.pid for r in recs if r.pid is not None}
            self._set(
                nvml_error=None,
                last_sample_ns=now,
                process_names={p: self._read_name(p) for p in sorted(pids)},
            )
        if self._dcgm_error() is not None:
            for metric in _DCGM_FIELDS.values():
                self._buffer.push(
                    unsupported(now, self._gpu.uuid, metric, "%", "dcgm", Scope.DEVICE)
                )
        self._refresh_mps(now)

    def _refresh_mps(self, now: int) -> None:
        last = self._mps_checked_ns
        if last is not None and 0 <= now - last < MPS_REFRESH_NS:
            return
        self._mps_checked_ns = now
        try:
            status = self._mps_query()
        except Exception:  # noqa: BLE001
            status = None  # unknown, not the previous answer
        self._set(mps=status)

    def _read_name(self, pid: int) -> str:
        try:
            with open(f"{self._proc_root}/{pid}/comm", encoding="utf-8", errors="replace") as f:
                name = f.read().strip()
        except OSError:
            return "?"
        return name or "?"
