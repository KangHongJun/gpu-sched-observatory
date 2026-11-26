"""``gso capture``: one on-demand Nsight Systems capture, then summary.json and timeline.txt."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

from gso.capability import detect
from gso.trace.nsys_capture import (
    MAX_DURATION_S,
    MIN_DURATION_S,
    CaptureError,
    CaptureUnavailable,
    capture,
)
from gso.trace.report import build_report
from gso.tui.widgets import NA

EXIT_USAGE = 2
EXIT_IO_ERROR = 3
EXIT_UNAVAILABLE = 4
EXIT_FAILED = 5
EXIT_INTERRUPTED = 130

_EPILOG = (
    f"Runs ONE on-demand capture of {MIN_DURATION_S}-{MAX_DURATION_S} s. Without a command after '--' it is a GLOBAL capture:\n"
    "context-switch events of every process on every GPU (the report covers --gpu only);\n"
    "no CUDA kernel intervals. With a command after '--' the command is launched under nsys\n"
    "and its kernel intervals are captured. For a launched command CUDA_DEVICE_ORDER=PCI_BUS_ID\n"
    "and CUDA_VISIBLE_DEVICES=<gpu> are set unless you already set them (the project's own\n"
    "workloads also set them themselves).\n"
    "The launched command is NOT ended when the capture ends unless --end-target is given.\n"
    "Never uses sudo; without root, other users' processes appear masked (summary: masked true).\n"
    "Writes DIR/summary.json, DIR/timeline.txt, DIR/capture.sqlite (and capture.nsys-rep with\n"
    "--keep-rep).\n"
    "exit codes:\n"
    "  0 done   2 usage error   3 I/O error writing the report   4 nsys not available\n"
    "  5 capture failed   130 interrupted"
)


def _duration(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if not MIN_DURATION_S <= v <= MAX_DURATION_S:
        raise argparse.ArgumentTypeError(
            f"must be {MIN_DURATION_S}..{MAX_DURATION_S}, got {text}"
        )
    return v


def add_parser(sub) -> None:
    p = sub.add_parser(
        "capture",
        help="run one on-demand nsys capture and write a summary and a text timeline",
        usage="gso capture --gpu N --out DIR [--duration D] [--end-target] [--keep-rep] [-- CMD ...]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument("--gpu", type=int, required=True, help="GPU index the report is about")
    p.add_argument("--duration", type=_duration, default=MAX_DURATION_S,
                   help=f"capture length in seconds, {MIN_DURATION_S}..{MAX_DURATION_S} "
                        f"(default {MAX_DURATION_S})")
    p.add_argument("--out", metavar="DIR", required=True, help="output directory (created if missing)")
    p.add_argument("--end-target", action="store_true",
                   help="end the launched command when the capture ends (needs a command after '--')")
    p.add_argument("--keep-rep", action="store_true",
                   help="keep capture.nsys-rep (deleted by default)")


def validate(parser: argparse.ArgumentParser, args, target: list[str] | None) -> None:
    """All checks happen before nsys runs; the output directory is created last."""
    if target is not None and not target:
        parser.error("nothing follows '--': give the command to launch")
    if args.end_target and target is None:
        parser.error("--end-target needs a command after '--'")
    if args.gpu < 0:
        parser.error("--gpu must be >= 0")
    out = Path(args.out)
    if out.exists() and not out.is_dir():
        parser.error(f"--out is not a directory: {args.out}")
    probe = out
    while not probe.exists():
        probe = probe.parent
    if not probe.is_dir():
        parser.error(f"cannot create --out below a non-directory: {probe}")
    if not os.access(probe, os.W_OK | os.X_OK):
        parser.error(f"directory is not writable: {probe}")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        parser.error(f"cannot create --out: {e}")
    args.target = target


def _target_env(gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    return env


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then ``os.replace``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _pct(x) -> str:
    return NA if x is None else f"{x * 100:.1f}%"


def _ms(x) -> str:
    return NA if x is None else f"{x:.3f} ms"


def format_summary_lines(s: dict) -> list[str]:
    """One line per key number; ``N/A`` for unknown; ``≥`` when the rate is a lower bound."""
    ctx = s["ctx"]
    rate = ctx["switch_rate_per_s"]
    if rate is None:
        rate_txt = NA
    else:
        rate_txt = f"{'≥ ' if ctx['switch_rate_is_lower_bound'] else ''}{rate:.1f} /s"
    win = s["window_s"]
    lines = [
        (f"GPU {s['gpu']}, mode {s['mode']}, window {NA if win is None else f'{win:.2f} s'}, "
         f"masked: {'yes' if s['masked'] else 'no'} (on-demand capture)"),
        f"context switch rate: {rate_txt}"
        + (" (lower bound: masked rows present)" if ctx["switch_rate_is_lower_bound"] else ""),
    ]
    for pc in ctx["per_context"]:
        r = pc["restores_per_s"]
        lines.append(
            f"context pid {pc['pid']}: restores {NA if r is None else f'{r:.1f}'} /s | "
            f"share restore_save {_pct(pc['restore_save']['share'])} (estimate), "
            f"hold {_pct(pc['hold']['share'])} (estimate)"
        )
    k = s["kernel"]
    count = NA if k["count"] is None else k["count"]
    lines.append(f"kernels: {count}, duration p50 {_ms(k['duration_ms']['p50_ms'])}")
    lines.append(
        f"launch-to-start p50 (CUDA API call start -> kernel start, estimate): "
        f"{_ms(k['launch_to_start_ms']['p50_ms'])}"
    )
    return lines


def run_capture(args) -> int:
    err = sys.stderr
    target = args.target
    try:
        caps = detect()
        res = capture(
            args.duration,
            Path(args.out),
            caps,
            target=target,
            name="capture",
            env=_target_env(args.gpu) if target is not None else None,
            end_target=args.end_target,
        )
        if not args.keep_rep:
            with contextlib.suppress(FileNotFoundError):
                res.rep_path.unlink()
        for w in res.warnings:
            print(f"gso: warning: {w}", file=err)
        summary, timeline = build_report(
            res.sqlite_path, gpu=args.gpu, mode=res.mode, masked=res.masked
        )
    except KeyboardInterrupt:
        print("gso: interrupted", file=err)
        return EXIT_INTERRUPTED
    except CaptureUnavailable as e:
        print(f"gso: {e}", file=err)
        return EXIT_UNAVAILABLE
    except CaptureError as e:
        print(f"gso: capture failed: {e}", file=err)
        return EXIT_FAILED
    except (ValueError, sqlite3.Error) as e:
        print(f"gso: could not read the capture: {e}", file=err)
        return EXIT_FAILED
    for line in format_summary_lines(summary):
        print(f"gso: {line}", file=err)
    out = Path(args.out)
    try:
        write_atomic(out / "summary.json",
                     json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        write_atomic(out / "timeline.txt", timeline)
    except OSError as e:
        print(f"gso: I/O error writing the report: {e}", file=err)
        return EXIT_IO_ERROR
    paths = [out / "summary.json", out / "timeline.txt", res.sqlite_path]
    if args.keep_rep:
        paths.append(res.rep_path)
    print("gso: wrote " + ", ".join(str(p) for p in paths), file=err)
    if res.mode == "launched" and not res.target_terminated:
        print("gso: the launched command was not ended by this capture and may still be "
              "running; it ends on its own", file=err)
    return 0
