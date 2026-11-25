"""DCGM provider: runs ``dcgmi dmon`` as a child process and feeds a RingBuffer.

Supplies the five device-level profiling counters. DCGM is optional: if it is missing, down
or a field is unsupported, records degrade to UNSUPPORTED/DROPPED and ``.error`` explains why.
Values from dcgmi are 0-1 fractions and are emitted as percent.

Each child runs a finite number of samples (about ``run_seconds``) and is respawned on a clean
exit, so with the container form a killed client can leave the in-container ``dcgmi dmon``
running for at most about ``run_seconds``.
"""

from __future__ import annotations

import collections
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
        run_seconds: float = 15.0,
    ) -> None:
        if interval_ms < _MIN_INTERVAL_MS:
            raise ValueError(f"interval_ms must be >= {_MIN_INTERVAL_MS}, got {interval_ms}")
        if not run_seconds > 0:
            raise ValueError(f"run_seconds must be > 0, got {run_seconds!r}")
        self.gpu_index = gpu_index
        self.gpu_uuid = gpu_uuid
        self._buffer = buffer
        self._requested = [i for i in sorted(FIELDS) if supported.get(i, False)]
        self._interval_ms = interval_ms
        self._count = max(1, math.ceil(run_seconds * 1000 / interval_ms))
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

    def _argv(self) -> list[str]:
        return [
            *self._cmd, "dmon", "-i", str(self.gpu_index),
            "-e", ",".join(str(i) for i in self._requested),
            "-d", str(self._interval_ms), "-c", str(self._count),
        ]

    def _spawn_locked(self):
        """Spawn a child; caller holds the lock. None if stopped or on spawn failure."""
        if self._stop.is_set():
            return None
        try:
            proc = self._popen(
                self._argv(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
            )
        except OSError as e:
            self.error = f"failed to start dcgmi: {e}"
            return None
        self._proc = proc
        return proc

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            if not self._requested:
                self.error = "no DCGM profiling fields are supported on this device"
                return
            self.error = None
            self._stop.clear()
            proc = self._spawn_locked()
            if proc is None:
                return
            self._thread = threading.Thread(
                target=self._run, args=(proc,), name="gso-dcgm", daemon=True
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

    @staticmethod
    def _reap(proc) -> None:
        """Terminate (then kill) and wait for a child, close its pipe. Never raises."""
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            pass  # already gone, or unkillable: nothing more we can do
        try:
            proc.stdout.close()
        except (OSError, ValueError, AttributeError):
            pass

    def _run(self, proc) -> None:
        try:
            while proc is not None:
                proc = self._consume(proc)
        except Exception as e:  # noqa: BLE001 - reader must never die silently
            if not self._stop.is_set():
                self.error = f"{type(e).__name__}: {e}"
            self._reap(proc if proc is not None else self._proc)

    def _consume(self, proc):
        """Read one child to its end. Returns the next child to read, or None to finish."""
        tail: collections.deque[str] = collections.deque(maxlen=5)
        rows = 0
        it = iter(proc.stdout)
        while True:
            try:
                line = next(it)
            except StopIteration:
                break
            except (OSError, ValueError):
                break  # pipe closed or undecodable stream
            row = parse_dmon_line(line, self._requested)
            if row is None:
                if line.strip() and not line.lstrip().startswith(("#", "ID")):
                    tail.append(line.strip())
                continue
            rows += 1
            self._emit(row)
        try:
            rc = proc.wait(timeout=2)
        except (subprocess.TimeoutExpired, OSError):
            rc = proc.poll()
        self._reap(proc)
        if self._stop.is_set():
            return None
        last = tail[-1] if tail else ""
        if rc == 0 and rows > 0:
            with self._lock:
                return self._spawn_locked()
        if rc == 0:
            msg = "dcgmi stream ended without data"
        else:
            msg = f"dcgmi exited with code {rc}"
        self.error = f"{msg}: {last}" if last else msg
        return None

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            proc, thread = self._proc, self._thread
        if proc is not None:
            self._reap(proc)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        if proc is not None:
            self._reap(proc)
