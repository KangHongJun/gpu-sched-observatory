"""On-demand capture job for the TUI worker thread (no Textual import).

``run_job`` never raises for capture problems: it returns the error text for the status line.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gso.capability import Capabilities
from gso.trace.nsys_capture import (
    CaptureCancelled,
    CaptureError,
    CaptureUnavailable,
    cancellable_runner,
    capture,
)
from gso.trace.report import build_report


@dataclass(frozen=True)
class CaptureReport:
    summary: dict  # summary for the watched GPU (see gso.trace.report.build_summary)
    timeline: str
    started_ns: int
    duration_s: int


@dataclass(frozen=True)
class CaptureOutcome:
    report: CaptureReport | None = None
    error: str | None = None


# (duration_s, out_dir, caps, gpu, cancel) -> CaptureReport; may raise CaptureUnavailable,
# CaptureError, CaptureCancelled. Runs in a worker thread and must stop soon after ``cancel`` is set.
CaptureRunner = Callable[[int, Path, Capabilities, int, threading.Event], CaptureReport]


def run_global_capture(
    duration_s: int, out_dir: Path, caps: Capabilities, gpu: int, cancel: threading.Event
) -> CaptureReport:
    """Real runner: a GLOBAL capture (no target), then the report for ``gpu``."""
    res = capture(duration_s, out_dir, caps, run=cancellable_runner(cancel), name="capture")
    if cancel.is_set():
        raise CaptureCancelled("capture cancelled")
    summary, timeline = build_report(res.sqlite_path, gpu=gpu, mode=res.mode, masked=res.masked)
    for p in (res.sqlite_path, res.rep_path):
        with contextlib.suppress(OSError):
            p.unlink()
    return CaptureReport(summary, timeline, res.started_ns, res.duration_s)


def _first_line(msg: str) -> str:
    return next((ln.strip() for ln in msg.splitlines() if ln.strip()), "unknown error")


def run_job(runner, duration_s, out_dir, caps, gpu, cancel) -> CaptureOutcome:
    try:
        return CaptureOutcome(report=runner(duration_s, out_dir, caps, gpu, cancel))
    except CaptureUnavailable as e:
        return CaptureOutcome(error=str(e))
    except CaptureCancelled:
        return CaptureOutcome(error="capture cancelled")
    except CaptureError as e:
        return CaptureOutcome(error=_first_line(str(e)))
    except (ValueError, OSError, sqlite3.Error) as e:
        return CaptureOutcome(error=f"capture report failed: {_first_line(str(e))}")
