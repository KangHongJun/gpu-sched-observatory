"""``gso bench``: the MPS A/B benchmark (two workloads together under five configurations)."""

from __future__ import annotations

import argparse
import contextlib
import math
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from gso.bench.profiling import default_profiler
from gso.bench.runner import (
    CONDITIONS,
    MIN_PROFILE_DURATION_S,
    MIN_REPS,
    Hooks,
    run_matrix,
)
from gso.bench.telemetry import NvmlDcgmTelemetry
from gso.bench.workloads import WORKLOADS
from gso.capability import detect

EXIT_USAGE = 2
EXIT_ABORTED = 3
EXIT_ERROR = 4
EXIT_INTERRUPTED = 130
_EXIT = {"complete": 0, "incomplete": 1, "aborted": EXIT_ABORTED, "error": EXIT_ERROR,
         "interrupted": EXIT_INTERRUPTED}
_RESERVED = ("--gpu", "--duration", "--out", "--start-at")  # the runner owns these

_EPILOG = (
    "Runs workloads A and B together on GPU N under: " + ", ".join(c[0] for c in CONDITIONS) + ".\n"
    "Per condition: one untimed warm-up, then --reps measured repetitions (>= 3). Both processes\n"
    "start measuring at the same wall-clock time (--start-at, setup budget --setup-budget); the\n"
    "GPU cools down between repetitions (max 60 s). Rows go to DIR/results.jsonl, run metadata to\n"
    "DIR/run.json. The MPS daemon is user-scoped, started per MPS condition and stopped after it;\n"
    "the run aborts if anything else appears on the GPU. Never uses sudo.\n"
    "exit codes: 0 complete  1 incomplete (failed repetitions)  2 usage error\n"
    "            3 aborted for safety  4 internal error  130 interrupted"
)


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return v


def add_parser(sub) -> None:
    p = sub.add_parser(
        "bench", help="MPS on/off A/B benchmark of two workloads on one GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=_EPILOG,
    )
    p.add_argument("--gpu", type=int, required=True, help="GPU index to run on (no default)")
    p.add_argument("--a", required=True, choices=WORKLOADS, metavar="WORKLOAD",
                   help="workload A: " + ", ".join(WORKLOADS))
    p.add_argument("--b", required=True, choices=WORKLOADS, metavar="WORKLOAD", help="workload B")
    p.add_argument("--out", required=True, metavar="DIR",
                   help="output directory; must not already contain results.jsonl")
    p.add_argument("--reps", type=int, default=MIN_REPS, help=f"repetitions per condition, >= {MIN_REPS} (default 3)")
    p.add_argument("--duration", type=_positive_float, default=30.0, help="measured seconds (default 30)")
    p.add_argument("--warmup", type=_positive_float, default=10.0,
                   help="untimed warm-up repetition seconds per condition (default 10)")
    p.add_argument("--conditions", nargs="+", metavar="NAME",
                   help="subset of: " + ", ".join(c[0] for c in CONDITIONS) + " (default all)")
    p.add_argument("--profile", action="store_true",
                   help=f"one 10 s global nsys capture per repetition (needs --duration >= {MIN_PROFILE_DURATION_S}); "
                        "profiled and unprofiled rows must not be compared")
    p.add_argument("--a-args", default="", metavar='"..."', help="extra arguments for workload A")
    p.add_argument("--b-args", default="", metavar='"..."', help="extra arguments for workload B")
    p.add_argument("--setup-budget", type=_positive_float, default=60.0, metavar="SECONDS",
                   help="time the workloads get for setup and warm-up before the common start (default 60)")
    p.add_argument("--dcgmi-cmd", default="dcgmi", help='dcgmi command, e.g. "docker exec gso-dcgm dcgmi"')
    p.add_argument("--dcgm-gpu-index", type=int, default=None,
                   help="GPU index passed to dcgmi -i (differs from the host index in a container)")


def validate(parser: argparse.ArgumentParser, args) -> None:
    """Every check happens before anything touches the GPU."""
    if args.gpu < 0:
        parser.error("--gpu must be >= 0")
    if args.reps < MIN_REPS:
        parser.error(f"--reps must be >= {MIN_REPS} (the spec requires at least 3 repetitions)")
    if args.profile and args.duration < MIN_PROFILE_DURATION_S:
        parser.error(f"--profile needs --duration >= {MIN_PROFILE_DURATION_S}")
    names = [n for item in (args.conditions or []) for n in item.split(",") if n]
    known = [c[0] for c in CONDITIONS]
    bad = [n for n in names if n not in known]
    if bad:
        parser.error(f"unknown condition(s): {', '.join(bad)}; choose from {', '.join(known)}")
    args.conditions = names or None
    for attr in ("a_args", "b_args"):
        try:
            toks = shlex.split(getattr(args, attr))
        except ValueError as e:
            parser.error(f"--{attr.replace('_', '-')}: {e}")
        # a token that equals OR abbreviates a reserved flag (--gp, --dur, --gpu=1 ...) is refused
        clash = [t for t in toks if t.startswith("--") and any(r.startswith(t.split("=")[0]) for r in _RESERVED)]
        if clash:
            parser.error(f"--{attr.replace('_', '-')} must not contain {', '.join(clash)} (set by the runner)")
        setattr(args, attr, toks)
    out = Path(args.out)
    if out.exists() and not out.is_dir():
        parser.error(f"--out is not a directory: {args.out}")
    if (out / "results.jsonl").exists():
        parser.error(f"{out / 'results.jsonl'} already exists; use a new --out so runs are never mixed")


def nsys_version(path: str) -> str | None:
    try:
        r = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10, check=False)
        return (r.stdout or "").strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _on_sigterm(signum, frame):
    raise KeyboardInterrupt


def run(args) -> int:
    err = sys.stderr
    dcgmi_cmd = shlex.split(args.dcgmi_cmd)
    caps = detect(dcgmi_cmd=dcgmi_cmd, dcgm_gpu_index=args.dcgm_gpu_index)
    if args.profile and not caps.nsys_path:
        print("gso bench: --profile needs nsys, which was not found (set NSYS or put nsys on PATH)", file=err)
        return EXIT_USAGE
    gpu = next((g for g in caps.gpus if g.index == args.gpu), None)
    if gpu is None:
        why = f" ({caps.nvml_error})" if caps.nvml_error else ""
        print(f"gso bench: GPU {args.gpu} was not found{why}; nothing was run", file=err)
        return EXIT_USAGE
    env_info = {
        "gpu_name": gpu.name if gpu else None,
        "driver_version": gpu.driver if gpu else None,
        "nsys_version": nsys_version(caps.nsys_path) if caps.nsys_path else None,
        "dcgm_available": caps.dcgm.available,
        "dcgm_reason": getattr(caps.dcgm, "reason", None),
    }
    hooks = Hooks(
        telemetry=NvmlDcgmTelemetry(
            args.gpu, caps.dcgm.fields if caps.dcgm.available else {},
            dcgmi_cmd=dcgmi_cmd, dcgm_gpu_index=args.dcgm_gpu_index),
        profiler=default_profiler(caps) if args.profile else None,
    )
    old = signal.signal(signal.SIGTERM, _on_sigterm)  # SIGTERM = Ctrl-C: clean up, exit 130
    try:
        outcome = run_matrix(
            args.gpu, args.a, args.b, Path(args.out), reps=args.reps, warmup_s=args.warmup,
            duration_s=args.duration, profile=args.profile, conditions=args.conditions,
            a_args=args.a_args, b_args=args.b_args, setup_budget_s=args.setup_budget,
            hooks=hooks, env_info=env_info)
    except ValueError as e:
        print(f"gso bench: {e}", file=err)
        return EXIT_USAGE
    finally:
        with contextlib.suppress(ValueError):
            signal.signal(signal.SIGTERM, old)
    detail = f": {outcome.message}" if outcome.message else ""
    print(f"gso bench: {outcome.status}{detail}; rows in {outcome.results_path}", file=err)
    return _EXIT[outcome.status]
