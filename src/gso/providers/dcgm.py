"""DCGM provider: runs ``dcgmi dmon`` as a child process and feeds a RingBuffer.

Supplies the five device-level profiling counters. DCGM is optional: if it is missing, down
or a field is unsupported, records degrade to UNSUPPORTED/DROPPED and ``.error`` explains why.
Values from dcgmi are 0-1 fractions and are emitted as percent.

Limitation: with the container form of ``dcgmi_cmd`` the real ``dcgmi dmon`` runs inside the
container and may outlive a killed ``docker exec`` client; this class cannot stop it.
"""

from __future__ import annotations

import math
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence

from gso.buffer import RingBuffer
from gso.record import Quality, Record, Scope, unsupported

FIELDS: dict[int, str] = {
    1001: "dcgm.gr_engine_active",
    1002: "dcgm.sm_active",
    1003: "dcgm.sm_occupancy",
    1004: "dcgm.tensor_active",
    1005: "dcgm.dram_active",
}
SOURCE = "dcgm"
_MIN_INTERVAL_MS = 100


def parse_dmon_line(line: str, field_ids: Sequence[int]) -> dict[int, float | None] | None:
    """Parse one ``dcgmi dmon`` line. None for headers, blanks and anything not a GPU row."""
    toks = line.split()
    if len(toks) < 2 or toks[0] != "GPU" or not toks[1].isdigit():
        return None
    values = toks[2:]
    out: dict[int, float | None] = {}
    for i, fid in enumerate(field_ids):
        v: float | None = None
        if i < len(values):
            try:
                v = float(values[i])
            except ValueError:
                v = None
            if v is not None and not math.isfinite(v):
                v = None
        out[fid] = v
    return out


class DcgmProvider:
    def __init__(
        self,
        gpu_index: int,
        gpu_uuid: str,
        buffer: RingBuffer,
        supported: Mapping[int, bool],
        interval_ms: int = 1000,
        dcgmi_cmd: Sequence[str] | None = None,
        popen=subprocess.Popen,
        clock_ns=time.time_ns,
    ) -> None:
        if interval_ms < _MIN_INTERVAL_MS:
            raise ValueError(f"interval_ms must be >= {_MIN_INTERVAL_MS}, got {interval_ms}")
        self.gpu_index = gpu_index
        self.gpu_uuid = gpu_uuid
        self._buffer = buffer
        self._requested = [i for i in sorted(FIELDS) if supported.get(i, False)]
        self._interval_ms = interval_ms
        self._cmd = list(dcgmi_cmd) if dcgmi_cmd is not None else ["dcgmi"]
        self._popen = popen
        self._clock_ns = clock_ns
        self._proc = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.error: str | None = None

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            if not self._requested:
                self.error = "no DCGM profiling fields are supported on this device"
                return
            argv = [
                *self._cmd, "dmon", "-i", str(self.gpu_index),
                "-e", ",".join(str(i) for i in self._requested),
                "-d", str(self._interval_ms),
            ]
            self.error = None
            self._stop.clear()
            try:
                proc = self._popen(
                    argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
            except OSError as e:
                self.error = f"failed to start dcgmi: {e}"
                return
            self._proc = proc
            self._thread = threading.Thread(
                target=self._read, args=(proc,), name="gso-dcgm", daemon=True
            )
            self._thread.start()

    def _emit(self, row: dict[int, float | None]) -> None:
        ts = self._clock_ns()
        for fid, metric in FIELDS.items():
            if fid not in row:
                rec = unsupported(ts, self.gpu_uuid, metric, "%", SOURCE, Scope.DEVICE)
            elif row[fid] is None:
                rec = Record(ts, self.gpu_uuid, metric, None, "%", SOURCE,
                             Scope.DEVICE, Quality.DROPPED)
            else:
                rec = Record(ts, self.gpu_uuid, metric, row[fid] * 100, "%", SOURCE,
                             Scope.DEVICE, Quality.MEASURED)
            self._buffer.push(rec)

    def _read(self, proc) -> None:
        first_line = ""
        try:
            for line in proc.stdout:
                row = parse_dmon_line(line, self._requested)
                if row is None:
                    if not first_line and line.strip() and not line.lstrip().startswith(("#", "ID")):
                        first_line = line.strip()
                    continue
                self._emit(row)
        except (OSError, ValueError):
            pass  # pipe closed under us (stop)
        try:
            rc = proc.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError):
            rc = proc.poll()
        if not self._stop.is_set():
            msg = f"dcgmi exited with code {rc}"
            if first_line:
                msg += f": {first_line}"
            self.error = msg

    def stop(self) -> None:
        self._stop.set()
        proc, thread = self._proc, self._thread
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except OSError:
                pass  # already gone
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
