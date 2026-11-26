"""One 10 s GLOBAL nsys capture for ``gso bench --profile`` (kernels are not captured)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from gso.capability import Capabilities
from gso.trace.nsys_capture import MAX_DURATION_S, CaptureError, capture
from gso.trace.report import build_summary

MPS_NOTE = "N/A under MPS: no context-switch events are recorded (decision 4)"


def default_profiler(caps: Capabilities):
    """``profiler(gpu, mps) -> {"ctx_switch_rate", "note", "error"}``. The capture runs under MPS
    too, so profiled repetitions carry the same capture load in every condition, but its context
    switch rate is not reported there."""

    def run(gpu: int, mps: bool) -> dict:
        with tempfile.TemporaryDirectory(prefix="gso-bench-nsys-") as d:
            try:
                res = capture(MAX_DURATION_S, Path(d), caps, name="capture")
                if mps:
                    return {"ctx_switch_rate": None, "note": MPS_NOTE, "error": None}
                summary = build_summary(
                    res.sqlite_path, gpu=gpu, mode="global", masked=res.masked
                )
            except (CaptureError, ValueError, OSError) as e:
                return {
                    "ctx_switch_rate": None,
                    "note": None,
                    "error": f"{type(e).__name__}: {e}",
                }
        ctx = summary["ctx"]
        note = (
            "lower bound: masked rows present"
            if ctx["switch_rate_is_lower_bound"]
            else None
        )
        return {
            "ctx_switch_rate": ctx["switch_rate_per_s"],
            "note": note,
            "error": None,
        }

    return run
