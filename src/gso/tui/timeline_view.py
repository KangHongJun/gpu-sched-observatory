"""F3 timeline text and the F4 confirmation screen."""

from __future__ import annotations

import time
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from gso.tui.capture_job import CaptureReport

NO_CAPTURE = "no capture yet — press F4"
THROUGHPUT_COST = (
    "measured cost: about -7 % throughput averaged over a 30 s run that contained "
    "the capture (one workload); the cost inside the capture window is larger and "
    "was not measured"
)
GLOBAL_SCOPE = (
    "the on-demand global capture records context switches for every GPU and "
    "every user's processes on this machine (the view shows GPU {gpu})"
)


def clock_text(started_ns: int) -> str:
    return time.strftime("%H:%M:%S", time.localtime(started_ns / 1e9))


def capture_label(report: CaptureReport) -> str:
    return f"capture {clock_text(report.started_ns)}, {report.duration_s} s"


def timeline_text(report: CaptureReport | None) -> str:
    if report is None:
        return NO_CAPTURE
    return f"on-demand capture ({capture_label(report)})\n{report.timeline}"


class ConfirmCaptureScreen(ModalScreen[bool]):
    DEFAULT_CSS = """
    ConfirmCaptureScreen { align: center middle; }
    #confirm-box { width: 70; height: auto; border: round $primary; padding: 1 2; background: $surface; }
    #confirm-buttons { height: 3; margin-top: 1; }
    """
    BINDINGS: ClassVar = [
        Binding("y", "confirm", show=False),
        Binding("n", "cancel", show=False),
        Binding("escape", "cancel", show=False),
    ]

    def __init__(self, duration_s: int, gpu: int) -> None:
        super().__init__()
        self._duration_s = duration_s
        self._gpu = gpu

    def compose(self) -> ComposeResult:
        msg = (
            f"on-demand capture, {self._duration_s} s, GLOBAL mode: "
            f"{GLOBAL_SCOPE.format(gpu=self._gpu)}. {THROUGHPUT_COST}."
        )
        with Vertical(id="confirm-box"):
            yield Static(msg, id="confirm-text", markup=False)
            with Horizontal(id="confirm-buttons"):
                yield Button("Capture", id="capture-ok", variant="primary")
                yield Button("Cancel", id="capture-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "capture-ok")

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
