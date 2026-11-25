"""Live metrics screen (F2). Reads only ``buffer.snapshot()`` and the collector's status."""

from __future__ import annotations

import time
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Static

from gso.buffer import RingBuffer
from gso.capability import Capabilities
from gso.record import Record
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


def _truncate(text: str, width: int) -> str:
    if width <= 0 or len(text) <= width:
        return text
    return text[: max(width - 3, 0)] + "..." if width > 3 else text[:width]


class GsoApp(App):
    CSS = """
    Screen { layout: vertical; }
    #header { height: 1; text-style: bold; }
    .bar { height: 1; }
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
        Binding("f4", "view('F4 Capture')", show=False),
        Binding("f5", "view('F5 Compare')", show=False),
    ]

    def __init__(
        self,
        buffer: RingBuffer,
        caps: Capabilities,
        gpu_index: int,
        collector=None,
        interval_s: float = 1.0,
        clock_ns=time.time_ns,
    ) -> None:
        super().__init__()
        self._buffer = buffer
        self._caps = caps
        self._gpu_index = gpu_index
        self._collector = collector
        self._interval_s = interval_s
        self._clock_ns = clock_ns
        self._max_age_ns = int(3 * interval_s * _S)
        self._notice: tuple[str, int] | None = None

    # -- layout -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Static("", id="header", markup=False)
        with Vertical(id="bars", classes="bars"):
            for wid, _, _ in _BARS:
                yield Static("", id=wid, classes="bar", markup=False)
        yield Static("", id="proc-table", markup=False)
        yield Static("", id="history", markup=False)
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

    # -- actions ----------------------------------------------------------
    def action_view(self, name: str) -> None:
        if name.startswith("F2"):
            self._notice = None
        else:
            self._notice = (f"{name}: not available yet", self._clock_ns() + NOTICE_S * _S)
        self.refresh_view()

    # -- rendering --------------------------------------------------------
    def _cstatus(self):
        return self._collector.status if self._collector is not None else None

    def _mps_state(self, cst) -> str:
        if cst is not None and cst.mps is not None:
            return "ON" if cst.mps.active else "OFF"
        if self._caps.mps_active:
            return "ON"
        return NA if cst is not None else "OFF"

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

        mps_state = self._mps_state(cst)
        header = " | ".join(
            [
                f"GPU {self._gpu_index}",
                self._gpu_name(cst),
                f"MPS {mps_state}",
                fmt_temp(val("gpu.temp")),
                fmt_power(val("gpu.power")),
            ]
        )
        self.query_one("#header", Static).update(header)
        for wid, label, metric in _BARS:
            self.query_one(f"#{wid}", Static).update(f"{label:<{_LABEL_W}}{bar(val(metric))}")

        self.query_one("#proc-table", Static).update(
            self._process_text(recs, cst, now, mps_state == "ON")
        )
        self.query_one("#history", Static).update(self._history_text(recs, now))
        self.query_one("#status", Static).update(self._status_text(cst, now))

    def _process_text(self, recs: list[Record], cst, now: int, mps_on: bool) -> str:
        names = dict(cst.process_names) if cst is not None else {}
        rows = process_rows(recs, names, now, self._max_age_ns)
        ctx = "N/A (MPS)" if mps_on else NA
        lines = [f"{'PID':<8}{'PROCESS':<18}{'SM UTIL':>8}{'VRAM':>8}  {'CTX SW/s'}"]
        for r in rows:
            sm = NA if r.sm_util is None else f"{r.sm_util:.0f}%"
            lines.append(
                f"{r.pid:<8}{r.name[:17]:<18}{sm:>8}{fmt_vram(r.vram_mib):>8}  {ctx}"
            )
        if not rows:
            seen = latest(recs, "gpu.util", None, now_ns=now, max_age_ns=self._max_age_ns)
            lines.append("(no GPU processes)" if seen is not None else NA)
        return "\n".join(lines)

    def _series(self, recs: list[Record], metric: str, now: int, width: int) -> list[float | None]:
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
        for label, metric in (("GPU Util", "gpu.util"), ("SM Active", "dcgm.sm_active")):
            line = sparkline(self._series(recs, metric, now, width), width)
            lines.append(f"{label:<{_LABEL_W}}{line}")
        return "\n".join(lines)

    def _status_text(self, cst, now: int) -> str:
        parts: list[str] = []
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
