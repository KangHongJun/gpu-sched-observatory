"""Live metrics screen (F2). Reads only ``buffer.snapshot()`` and the collector's status."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Static
from textual.worker import Worker, WorkerState

from gso.bench import report as bench_report
from gso.buffer import RingBuffer
from gso.capability import Capabilities
from gso.record import Record
from gso.tui.capture_job import (
    CaptureReport,
    CaptureRunner,
    run_global_capture,
    run_job,
)
from gso.tui.timeline_view import ConfirmCaptureScreen, capture_label, timeline_text
from gso.tui.widgets import (
    NA,
    bar,
    fmt_power,
    fmt_temp,
    fmt_vram,
    latest,
    process_rows,
    sparkline,
)

# /proc/<pid>/comm is cut to 15 chars, so the MPS server appears as "nvidia-cuda-mps".
# The full name matches too. nvidia-cuda-mps-control truncates to the same prefix, but it
# never holds a compute context on the watched GPU, so it cannot cause a false ON.
MPS_COMM_PREFIX = "nvidia-cuda-mps"
_S = 1_000_000_000
HISTORY_S = 60
NOTICE_S = 5
FOOTER = "F2 Metrics | F3 Timeline | F4 Capture | F5 Compare"
_BARS = [
    ("gr-engine", "GR Engine", "dcgm.gr_engine_active"),
    ("sm-active", "SM Active", "dcgm.sm_active"),
    ("sm-occupancy", "SM Occupancy", "dcgm.sm_occupancy"),
    ("tensor-active", "Tensor Active", "dcgm.tensor_active"),
    ("dram-active", "DRAM Active", "dcgm.dram_active"),
    ("nvdec", "NVDEC", "gpu.dec_util"),
]
_LABEL_W = 14
EXIT_WAIT_S = 15
CAPTURING = "캡처 중"  # DCGM values stay visible but are marked while nsys runs (RESULTS, Task 8)


def _truncate(text: str, width: int) -> str:
    if width <= 0 or len(text) <= width:
        return text
    return text[: max(width - 3, 0)] + "..." if width > 3 else text[:width]


class GsoApp(App):
    CSS = """
    Screen { layout: vertical; }
    #header { height: 1; text-style: bold; }
    .bar { height: 1; }
    #metrics-view { height: 1fr; }
    #timeline-view { height: 1fr; display: none; overflow: hidden; }
    #compare-view { height: 1fr; display: none; overflow-x: auto; overflow-y: auto; }
    #compare-text { width: auto; height: auto; text-wrap: nowrap; text-overflow: clip; }
    #proc-table { height: 1fr; min-height: 2; margin-top: 1; overflow: hidden; }
    #history { height: 4; }
    #status { height: 1; }
    #footer { height: 1; }
    """
    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit", show=False),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("f2", "view('F2 Metrics')", show=False),
        Binding("f3", "view('F3 Timeline')", show=False),
        Binding("f4", "capture", show=False),
        Binding("f5", "view('F5 Compare')", show=False),
    ]

    def __init__(
        self,
        buffer: RingBuffer,
        caps: Capabilities,
        gpu_index: int,
        collector=None,
        interval_s: float = 1.0,
        dcgm_interval_ms: int = 1000,
        clock_ns=time.time_ns,
        capture_runner: CaptureRunner | None = None,
        capture_duration_s: int = 10,
        bench_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._buffer = buffer
        self._caps = caps
        self._gpu_index = gpu_index
        self._collector = collector
        self._interval_s = interval_s
        self._clock_ns = clock_ns
        # a value is stale after three missed samples of the slower of the two sources
        self._max_age_ns = int(3 * max(interval_s, dcgm_interval_ms / 1000) * _S)
        self._notice: tuple[str, int] | None = None
        self._runner = capture_runner or run_global_capture
        self._capture_duration_s = capture_duration_s
        self._capture_report: CaptureReport | None = None
        self._capture_running = False
        self._capture_started = 0.0
        self._capture_error: str | None = None
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._done.set()
        self._tmpdir: Path | None = None
        self._quitting = False
        self._view = "F2"
        self._compare_text = bench_report.view_text(bench_dir)  # static: read once

    @property
    def buffer(self) -> RingBuffer:
        return self._buffer

    @property
    def collector(self):
        return self._collector

    # -- layout -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static("", id="header", markup=False)
        with Vertical(id="metrics-view"):
            with Vertical(id="bars", classes="bars"):
                for wid, _, _ in _BARS:
                    yield Static("", id=wid, classes="bar", markup=False)
            yield Static("", id="proc-table", markup=False)
            yield Static("", id="history", markup=False)
        yield Static("", id="timeline-view", markup=False)
        with VerticalScroll(id="compare-view"):
            yield Static(self._compare_text, id="compare-text", markup=False)
        yield Static("", id="status", markup=False)
        yield Static(FOOTER, id="footer", markup=False)

    def on_mount(self) -> None:
        if self._collector is not None:
            self._collector.start()
        self.refresh_view()
        self.set_interval(self._interval_s, self.refresh_view)

    async def on_unmount(self) -> None:
        if self._collector is not None:
            self._collector.stop()
        self._cancel.set()  # ends a running capture through the wrapper's own cancel path
        await asyncio.to_thread(self._done.wait, EXIT_WAIT_S)
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # -- actions ----------------------------------------------------------
    def _set_notice(self, text: str) -> None:
        self._notice = (text, self._clock_ns() + NOTICE_S * _S)

    def action_view(self, name: str) -> None:
        if name.startswith(("F2", "F3", "F5")):
            self._notice = None
            self._view = name[:2]
            self._show_view()
        else:
            self._set_notice(f"{name}: not available yet")
        self.refresh_view()

    def _show_view(self) -> None:
        self.query_one("#metrics-view").display = self._view == "F2"
        self.query_one("#timeline-view").display = self._view == "F3"
        self.query_one("#compare-view").display = self._view == "F5"

    def action_capture(self) -> None:
        if self._capture_running:
            self._set_notice("capture already running")
            self.refresh_view()
        elif not isinstance(self.screen, ConfirmCaptureScreen):
            self.push_screen(
                ConfirmCaptureScreen(self._capture_duration_s, self._gpu_index),
                self._confirmed,
            )

    def _confirmed(self, ok: bool | None) -> None:
        if not ok or self._capture_running:
            return
        if self._tmpdir is None:
            self._tmpdir = Path(tempfile.mkdtemp(prefix="gso-capture-"))
        self._capture_error = None
        self._capture_running = True
        self._capture_started = time.monotonic()
        self._cancel.clear()
        self._done.clear()
        self.run_worker(
            self._capture_job, thread=True, name="capture", exit_on_error=False
        )
        self.refresh_view()

    def _capture_job(self):
        # worker thread: nsys, SQLite loading and metric computation all happen here
        try:
            return run_job(
                self._runner,
                self._capture_duration_s,
                self._tmpdir,
                self._caps,
                self._gpu_index,
                self._cancel,
            )
        finally:
            self._done.set()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "capture" or event.state not in (
            WorkerState.SUCCESS,
            WorkerState.ERROR,
            WorkerState.CANCELLED,
        ):
            return
        outcome = event.worker.result if event.state is WorkerState.SUCCESS else None
        self._capture_running = False
        if outcome is not None and outcome.report is not None:
            self._capture_report = outcome.report
        elif outcome is not None:
            self._capture_error = outcome.error
        elif event.state is WorkerState.ERROR:
            self._capture_error = f"capture failed: {event.worker.error}"
        if self.is_mounted:
            self.refresh_view()

    async def action_quit(self) -> None:
        if self._capture_running:
            elapsed = time.monotonic() - self._capture_started
            self._quitting = True
            self.refresh_view()
            await asyncio.sleep(0.05)  # let the notice paint
            self._cancel.set()
            wait = max(self._capture_duration_s - elapsed, 0) + EXIT_WAIT_S
            await asyncio.to_thread(self._done.wait, wait)
        self.exit()

    # -- rendering --------------------------------------------------------
    def _cstatus(self):
        return self._collector.status if self._collector is not None else None

    def _mps_state(self, rows, cst) -> str:
        """Per watched GPU, from its NVML process list: ON when any process there is an MPS
        client or the MPS server. The host-wide control-pipe answer is not the GPU's state."""
        if cst is not None and cst.nvml_error is not None:
            return NA
        names = cst.process_names if cst is not None else {}
        for r in rows:
            if r.is_mps or names.get(r.pid, "").startswith(MPS_COMM_PREFIX):
                return "ON"
        return "OFF"

    def _mps_host_daemon(self, cst) -> bool:
        if cst is not None and cst.mps is not None:
            return cst.mps.active
        return self._caps.mps_active

    def _gpu_name(self, cst) -> str:
        if cst is not None and cst.gpu_name:
            return cst.gpu_name
        for g in self._caps.gpus:
            if g.index == self._gpu_index:
                return g.name
        return NA

    def refresh_view(self) -> None:
        now = self._clock_ns()
        recs = self._buffer.snapshot(since_ns=now - (HISTORY_S + 2) * _S)
        cst = self._cstatus()
        age = self._max_age_ns

        def val(metric: str) -> float | None:
            r = latest(recs, metric, None, now_ns=now, max_age_ns=age)
            return None if r is None else r.value

        names = dict(cst.process_names) if cst is not None else {}
        rows = process_rows(recs, names, now, self._max_age_ns)
        mps_state = self._mps_state(rows, cst)
        mps_text = f"MPS {mps_state}"
        if mps_state != "ON" and self._mps_host_daemon(cst):
            mps_text += " (MPS daemon on host)"
        header = " | ".join(
            [
                f"GPU {self._gpu_index}",
                self._gpu_name(cst),
                mps_text,
                fmt_temp(val("gpu.temp")),
                fmt_power(val("gpu.power")),
            ]
        )
        self.query_one("#header", Static).update(header)
        mark = f" {CAPTURING}" if self._capture_running else ""
        for wid, label, metric in _BARS:
            tag = mark if metric.startswith("dcgm.") else ""
            self.query_one(f"#{wid}", Static).update(
                f"{label:<{_LABEL_W}}{bar(val(metric))}{tag}"
            )
        self.query_one("#timeline-view", Static).update(
            timeline_text(self._capture_report)
        )

        self.query_one("#proc-table", Static).update(
            self._process_text(rows, recs, now, mps_state == "ON")
        )
        self.query_one("#history", Static).update(self._history_text(recs, now))
        self.query_one("#status", Static).update(self._status_text(cst, now))

    def _process_text(self, rows, recs: list[Record], now: int, mps_on: bool) -> str:
        rates = self._ctx_rates() if not mps_on else {}
        note = ""
        if self._capture_report is not None and not mps_on:
            note = f" ({capture_label(self._capture_report)})"
        lines = [f"{'PID':<8}{'PROCESS':<18}{'SM UTIL':>8}{'VRAM':>8}  CTX SW/s{note}"]
        for r in rows:
            sm = NA if r.sm_util is None else f"{r.sm_util:.0f}%"
            if mps_on:
                ctx = "N/A (MPS)"
            else:
                rate = rates.get(r.pid)
                ctx = NA if rate is None else f"{rate:.1f}"
            lines.append(
                f"{r.pid:<8}{r.name[:17]:<18}{sm:>8}{fmt_vram(r.vram_mib):>8}  {ctx}"
            )
        if not rows:
            seen = latest(
                recs, "gpu.util", None, now_ns=now, max_age_ns=self._max_age_ns
            )
            lines.append("(no GPU processes)" if seen is not None else NA)
        return "\n".join(lines)

    def _ctx_rates(self) -> dict[int, float | None]:
        if self._capture_report is None:
            return {}
        pcs = self._capture_report.summary.get("ctx", {}).get("per_context", [])
        return {
            pc["pid"]: pc["restores_per_s"] for pc in pcs if pc.get("pid") is not None
        }

    def _series(
        self, recs: list[Record], metric: str, now: int, width: int
    ) -> list[float | None]:
        out: list[float | None] = [None] * width
        newest = [-1] * width
        for r in recs:
            if r.metric != metric or r.pid is not None:
                continue
            b = (now - r.ts_ns) // _S
            if 0 <= b < width and r.ts_ns > newest[b]:
                newest[b] = r.ts_ns
                out[b] = r.value
        out.reverse()  # oldest first
        return out

    def _history_text(self, recs: list[Record], now: int) -> str:
        width = max(10, min(HISTORY_S, self.size.width - _LABEL_W - 1))
        lines = [f"History (last {width} s)"]
        for label, metric in (
            ("GPU Util", "gpu.util"),
            ("SM Active", "dcgm.sm_active"),
        ):
            line = sparkline(self._series(recs, metric, now, width), width)
            lines.append(f"{label:<{_LABEL_W}}{line}")
        return "\n".join(lines)

    def _status_text(self, cst, now: int) -> str:
        parts: list[str] = []
        if self._quitting:
            parts.append("waiting for capture to finish…")
        elif self._capture_running:
            parts.append(
                f"capture running… {int(time.monotonic() - self._capture_started)} s"
            )
        elif self._capture_error:
            parts.append(f"capture: {self._capture_error}")
        if self._notice is not None and now < self._notice[1]:
            parts.append(self._notice[0])
        dcgm_err = None
        if cst is not None:
            dcgm_err = cst.dcgm_error
        elif not self._caps.dcgm.available:
            dcgm_err = self._caps.dcgm.reason or "unavailable"
        if dcgm_err:
            parts.append(f"DCGM: {dcgm_err}")
        nvml_err = cst.nvml_error if cst is not None else self._caps.nvml_error
        if nvml_err:
            parts.append(f"NVML: {nvml_err}")
        if self._buffer.dropped:
            parts.append(f"dropped: {self._buffer.dropped}")
        return _truncate(" | ".join(parts), self.size.width)
