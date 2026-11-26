"""Launch workload processes together, wait for them, terminate exactly the groups we made.

Each process is started with ``start_new_session=True`` so it leads its own process group
(pgid == pid). On an error, Ctrl-C or SIGTERM of the runner, ``terminate`` signals only those
groups, and only while their leader is still running (so a recycled PID is never signalled),
SIGTERM first and SIGKILL after a grace period. Nothing is ever looked up by name.

Start barrier design (one of the two the task allowed): the runner computes
``start_at = now + setup_budget_s`` and passes the SAME value as ``--start-at`` to every process.
Each workload does its own setup and warm-up, sleeps until ``start_at`` and then measures. It was
chosen over a READY handshake because it needs no protocol between runner and workload (no stdout
parsing, no stdin) and cannot deadlock; a process whose setup outlasts the budget exits 1 with
"start barrier missed" and the repetition is recorded as failed, loudly. The price is that every
repetition lasts ``setup_budget_s + duration``; ``--setup-budget`` lowers it when setup is fast.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

TERM_GRACE_S = 10.0
POLL_S = 0.2
DEADLINE_GRACE_S = 60.0  # after the window should have ended: result writing and process exit
_TAIL_CHARS = 300
_NEEDED = ("throughput_per_s", "latency_ms", "metric_kind")


@dataclass
class ProcSpec:
    label: str
    workload: str
    argv: list[str]
    env: dict[str, str]
    out_path: Path


@dataclass
class ProcResult:
    label: str
    workload: str
    exit_code: int | None
    payload: dict | None
    error: str | None
    stderr_tail: str


def workload_argv(
    workload: str, gpu_arg: int, duration_s: float, out: Path, start_at: float | None,
    extra: Sequence[str],
) -> list[str]:
    import sys

    argv = [sys.executable, "-m", f"gso.bench.workloads.{workload}", "--gpu", str(gpu_arg),
            "--duration", f"{duration_s:g}", "--out", str(out)]
    if start_at is not None:
        argv += ["--start-at", f"{start_at:.3f}"]
    return argv + list(extra)


def window_overlap_s(payloads: Sequence[dict | None]) -> float | None:
    """Seconds during which every measured window was open, or None if any window is missing."""
    spans = []
    for p in payloads:
        if not p or p.get("window_start_ns") is None or p.get("window_end_ns") is None:
            return None
        spans.append((p["window_start_ns"], p["window_end_ns"]))
    if not spans:
        return None
    return max(0.0, (min(e for _, e in spans) - max(s for s, _ in spans)) / 1e9)


class Group:
    """One launched process and the temp file its stderr goes to."""

    def __init__(self, spec: ProcSpec, proc, err):
        self.spec, self.proc, self.err = spec, proc, err
        self.note: str | None = None


def _tail(err) -> str:
    try:
        err.flush()
        err.seek(0)
        text = err.read().decode("utf-8", "replace")
    except (OSError, ValueError):
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " | ".join(lines[-3:])[-_TAIL_CHARS:]


def terminate(groups: Sequence[Group], killpg=os.killpg, grace_s: float = TERM_GRACE_S) -> None:
    """SIGTERM then SIGKILL the process groups of still-running processes we launched."""
    running = []
    for g in groups:
        pid = g.proc.pid
        if g.proc.poll() is not None or not isinstance(pid, int) or pid <= 1 or pid == os.getpid():
            continue
        with contextlib.suppress(ProcessLookupError, PermissionError):
            killpg(pid, signal.SIGTERM)
        running.append(g)
    for g in running:
        try:
            g.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                killpg(g.proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                g.proc.wait(timeout=grace_s)


def _result(g: Group) -> ProcResult:
    code = g.proc.returncode
    tail = _tail(g.err)
    payload, error = None, g.note
    if error is None and code == 0:
        try:
            payload = json.loads(g.spec.out_path.read_text())
            missing = [k for k in _NEEDED if not isinstance(payload, dict) or k not in payload]
            if missing:
                payload, error = None, f"result JSON lacks {', '.join(missing)}"
        except (OSError, ValueError) as e:
            payload, error = None, f"no usable result JSON ({type(e).__name__})"
    elif error is None:
        error = f"exit {code}" + (f": {tail}" if tail else "")
    return ProcResult(g.spec.label, g.spec.workload, code, payload, error, tail)


def run_procs(
    specs: Sequence[ProcSpec],
    *,
    launch=subprocess.Popen,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    deadline_at: float,
    actions: Sequence[tuple[float, Callable[[], None]]] = (),
    killpg=os.killpg,
) -> list[ProcResult]:
    """Start every spec, run the timed ``actions`` (epoch seconds, blocking callables) while
    waiting, and return one result per spec. If any process fails while another still runs the
    others are terminated (the repetition is lost anyway). Past ``deadline_at`` everything still
    running is terminated and reported as timed out. Interrupts terminate our groups and re-raise.
    """
    groups: list[Group] = []
    pending = sorted(actions, key=lambda a: a[0])
    try:
        with contextlib.ExitStack() as stack:
            for spec in specs:
                err = stack.enter_context(tempfile.TemporaryFile())
                proc = launch(
                    spec.argv, env=spec.env, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=err, start_new_session=True,
                )
                groups.append(Group(spec, proc, err))
            while True:
                codes = [g.proc.poll() for g in groups]
                if all(c is not None for c in codes):
                    break
                if any(c not in (None, 0) for c in codes):
                    for g, c in zip(groups, codes, strict=True):
                        if c is None:
                            g.note = "terminated: a peer process failed"
                    terminate(groups, killpg)
                    break
                now = clock()
                if now >= deadline_at:
                    for g, c in zip(groups, codes, strict=True):
                        if c is None:
                            g.note = "timed out: still running after the deadline"
                    terminate(groups, killpg)
                    break
                if pending and now >= pending[0][0]:
                    pending.pop(0)[1]()
                    continue
                sleep(POLL_S)
            return [_result(g) for g in groups]
    except BaseException:
        terminate(groups, killpg)
        raise
